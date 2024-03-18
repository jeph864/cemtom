from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Dirichlet, LogNormal, LogisticNormal, kl_divergence
import numpy as np
import datetime

from .distributions import VariationalDistribution

torch.manual_seed(434)


def init_logistic_prior(alpha):
    pass


class InferenceNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=(100, 100), activation='relu', dropout=0.20,
                 batch_norm=True,
                 affine=False, *args,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        assert isinstance(in_features, int), "input_size must by type int."
        assert isinstance(out_features, int), "output_size must be type int."
        assert isinstance(hidden_dims, tuple), "hidden_sizes must be type tuple."
        assert dropout >= 0, "dropout must be >= 0."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()

        self.input = nn.Linear(self.in_features, self.hidden_dims[0])
        hidden_layers = []

        for i, (hidden_in, hidden_out) in enumerate(zip(self.hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append((f'layer_{i}', nn.Sequential(nn.Linear(hidden_in, hidden_out), self.activation)))
        self.hidden_layers = nn.Sequential(OrderedDict(hidden_layers))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_bert=None):
        hidden = self.hidden_layers(self.activation(self.input(x)))
        hidden = self.dropout(hidden)
        return hidden


class ContextualizedInferenceNet(InferenceNet):
    def __init__(self, in_features, out_features, embeddings_size, hidden_dims=(100, 100), activation='relu',
                 dropout=0.20):
        super().__init__(in_features, out_features, hidden_dims=hidden_dims, activation=activation, dropout=dropout)
        self.embeddings_size = embeddings_size
        self.adapt_embeddings = nn.Linear(self.embeddings_size, self.in_features)
        self.input = nn.Linear(self.in_features * 2, hidden_dims[0])

    def forward(self, x, x_bert=None):
        x_bert = self.adapt_embeddings(x_bert)
        x = torch.cat((x, x_bert), 1)
        return super().forward(x)


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, dropout=0.2):
        super().__init__()
        self.beta = torch.Tensor(latent_size, output_size)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(num_features=output_size, affine=False)
        self.dropout_theta = nn.Dropout(p=dropout)
        self.topic_word_matrix = None

    def forward(self, theta):
        theta = self.dropout_theta(theta)
        # ProdLDA
        recon_x = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        topic_word = self.beta
        self.topic_word_matrix = self.beta
        return recon_x, topic_word


class ContextualizedDecoder(nn.Module):
    def __init__(self, latent_size, vocab_size, embedding_size, embeddings=None, train_embeddings=True,
                 p_dropout=0.5):
        super().__init__()
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout = p_dropout
        if train_embeddings:
            self.word_embeddings = nn.Linear(self.embedding_size, vocab_size, bias=False)
        else:
            num_embeddings, embedding_size = embeddings.size()
            rho = nn.Embedding(num_embeddings, embedding_size)
            self.word_embeddings = embeddings.clone().float()
            if torch.cuda.is_available():
                self.word_embeddings.cuda()
        self.topic_embeddings = nn.Linear(embedding_size, latent_size, bias=False)
        # self.dropout = nn.Dropout(p_dropout)

    def get_beta(self):
        if hasattr(self.word_embeddings, 'weight'):
            logit = self.topic_embeddings(self.word_embeddings.weight)
        else:
            logit = self.topic_embeddings(self.word_embeddings)

        beta = F.softmax(logit, dim=0).transpose(1, 0)
        return beta

    def forward(self, theta):

        beta = self.get_beta()
        res = torch.mm(theta, beta)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        x_recon = results_without_zeros  # torch.log(results_without_zeros)
        return x_recon, beta


class FCDecoder(nn.Module):
    def __init__(self, topic_size, vocab_size, dropout=0.2):
        super().__init__()

        self.beta = nn.Linear(topic_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(vocab_size)
        self.batch_norm.weight.requires_grad = False

    def forward(self, theta):
        theta = self.dropout(theta)
        recon_x = self.batch_norm(self.beta(theta))  # F.softmax(, dim=1)
        return F.softmax(recon_x, -1), self.beta.weight.T


class BaseVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, h2dist=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.h2dist = h2dist
        self.document_topic_matrix = None
        self.topic_word_matrix = None

    def forward(self, x, x_bert=None):
        hidden = self.encoder(x, x_bert)
        z, dist = self.h2dist(hidden)
        recon_x, topic_words = self.decoder(z)
        return recon_x, hidden, z, dist, topic_words

    def get_theta(self):
        pass

    def sample(self):
        pass

    def get_beta(self, normalized=True):
        self.eval()
        # Assuming beta is a parameter of the decoder
        with torch.no_grad():
            if hasattr(self.decoder, 'beta'):
                if not normalized:
                    beta = self.decoder.beta
                else:
                    beta = F.softmax(self.decoder.beta, dim=1)
            else:
                raise AttributeError("Decoder does not have attribute 'beta'.")
        return beta


class ProdLDA(BaseVAE):
    def __init__(self, vocab_size, topic_size, hidden_dims=(100, 100), dropout=0.2,
                 prior_mean=0.0, prior_variance=None, alpha=40, learn_priors=True):
        encoder = InferenceNet(vocab_size, topic_size, hidden_dims=hidden_dims, activation='softplus', dropout=dropout)
        decoder = FCDecoder(topic_size, vocab_size, dropout=dropout)
        h2dist = H2LogisticNormal(hidden_dims[-1], topic_size)
        super().__init__(encoder, decoder, h2dist)
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_dims = hidden_dims

        # Move Later ???
        self.prior_mean = torch.tensor(
            [prior_mean] * topic_size)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if prior_variance is None:
            prior_variance = (1 / alpha) * (
                        1 - (2 / self.topic_size) + (1 / (self.topic_size * alpha)))  # 1. - (1. / self.topic_size)
        self.prior_variance = torch.tensor(
            [prior_variance] * topic_size)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.learn_priors = learn_priors
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

    def compute_loss(self, x=None, recon_x=None,
                     posterior=None, **args):
        posterior_variance = posterior.scale
        posterior_mean = posterior.loc
        posterior_log_variance = 2 * torch.log(posterior.scale)
        """latent_loss = 0.5 * (torch.sum(torch.div(posterior_variance, self.prior_variance), 1) +
                             torch.sum(
                                 (self.prior_mean - posterior_mean).pow(2) / self.prior_variance, dim=1)
                             - self.topic_size +
                             torch.sum(torch.log(self.prior_variance)) - torch.sum(posterior_log_variance, dim=1)
                             )"""
        # var division term
        var_division = torch.sum(posterior_variance / self.prior_variance, dim=1)
        # diff means term
        diff_means = self.prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / self.prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            self.prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        latent_loss = 0.5 * (var_division + diff_term - self.topic_size + logvar_det_division)

        # Reconstruction term
        recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1)
        return latent_loss, recon_loss


class ContextualizedProdLDA(ProdLDA):
    def __init__(self, vocab_size, topic_size, embedding_size, hidden_dims=(100, 100), dropout=0.2, alpha=60.0,
                 prior_mean=0.0, prior_variance=None, learn_priors=True):
        super().__init__(vocab_size, topic_size, hidden_dims, dropout, prior_mean, prior_variance, alpha, learn_priors)
        self.encoder = ContextualizedInferenceNet(self.vocab_size, self.topic_size,
                                                  embedding_size, hidden_dims=self.hidden_dims,
                                                  activation='softplus', dropout=dropout
                                                  )


class ZeroShotProdLDA(ProdLDA):
    pass


class ETM(BaseVAE):
    def __init__(self, vocab_size, topic_size, embedding_size, embeddings=None,
                 hidden_dims=(100,), activation='softplus', train_embeddings=True,
                 p_dropout=0.5):
        encoder = InferenceNet(vocab_size, topic_size, hidden_dims=hidden_dims,
                               activation=activation, dropout=p_dropout
                               )
        decoder = ContextualizedDecoder(topic_size, vocab_size, embedding_size, embeddings,
                                        train_embeddings=train_embeddings, p_dropout=p_dropout)
        h2dist = H2LogisticNormal(hidden_dims[-1], topic_size)
        super().__init__(encoder, decoder, h2dist)
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_dims = hidden_dims

    def forward(self, x, x_norm=None):
        hidden = self.encoder(x)
        z, dist = self.h2dist(hidden)
        recon_x, topic_words = self.decoder(z)
        return recon_x, hidden, z, dist, topic_words

    def compute_loss(self, x=None, recon_x=None,
                     posterior=None):
        posterior_variance = posterior.scale
        posterior_mean = posterior.loc
        posterior_log_variance = torch.log(posterior.scale)
        prior_loc = torch.zeros_like(posterior.loc)
        prior_scale = torch.ones_like(posterior.scale)
        latent_loss = 0.5 * (torch.sum(torch.div(posterior_variance, prior_scale), 1) +
                             torch.sum(
                                 (prior_loc - posterior_mean).pow(2) / prior_scale, dim=1)
                             - self.topic_size +
                             torch.sum(torch.log(prior_scale)) - torch.sum(posterior_log_variance, dim=1)
                             )
        # Reconstruction term
        recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1)

        return latent_loss, recon_loss


class H2Normal(nn.Module):
    def __init__(self, hidden_dim, latent_dim, prior_mean=0.0, prior_variance=None, learn_priors=False, batch_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.f_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.f_mu_batchnorm = nn.BatchNorm1d(self.latent_dim, affine=False)
        self.f_sigma = nn.Linear(self.hidden_dim, self.latent_dim)
        self.f_sigma_batchnorm = nn.BatchNorm1d(self.latent_dim, affine=False)
        # priors
        self.prior_mean = torch.tensor(
            [prior_mean] * latent_dim)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        self.prior_variance = torch.tensor(
            [prior_variance] * latent_dim)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.learn_priors = learn_priors
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        return z, dist

    def forward(self, hidden):
        mu = self.f_mu_batchnorm(self.f_mu(hidden))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(hidden))
        return self.reparameterize(mu, log_sigma)


class H2LogNormal(nn.Module):
    pass


class H2LogisticNormal(H2Normal):
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        dist = Normal(mu, std)

        if self.training:
            eps = torch.randn_like(std)
            # z = dist.rsample()
            z = eps.mul(std).add_(mu)
        else:
            z = dist.mean
            z = mu
        return F.softmax(z, dim=1), dist


class H2Dirichlet(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn = nn.BatchNorm1d(self.latent_dim)

    def forward(self, hidden):
        alphas = self.bn(self.fc(hidden))
        dist = Dirichlet(alphas)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        return z, dist

    @staticmethod
    def kl(posterior, prior):
        return torch.distributions.kl_divergence(posterior, prior).mean()
