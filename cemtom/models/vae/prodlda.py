from types import SimpleNamespace

from torch.distributions import Normal, LogNormal

from .distributions import LogisticNormalDistribution
from .vae import BaseVAE, Decoder, InferenceNet

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_normal_normal(posterior: Normal, prior: Normal):
    posterior_variance = posterior.scale
    posterior_mean = posterior.loc
    posterior_log_variance = 2 * torch.log(posterior.scale)
    topic_size = prior.mean.size(0)
    return 0.5 * (torch.sum(torch.div(posterior_variance, prior.scale), 1) +
                  torch.sum(
                      (prior.mean - posterior_mean).pow(2) / prior.scale, dim=1)
                  - topic_size +
                  torch.sum(torch.log(prior.scale)) - torch.sum(posterior_log_variance, dim=1)
                  )


def kl_normal_normal_true(posterior, prior):
    pass


class ProdLDADecoder(Decoder):
    def __init__(self, topic_size, vocab_size, dropout=0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.Tensor(topic_size, vocab_size))
        nn.init.xavier_uniform_(self.beta)
        self.norm = nn.BatchNorm1d(num_features=vocab_size, affine=False)
        self.dropout = nn.Dropout(p=dropout)

    def decode(self, theta):
        theta = self.dropout(theta)
        word_dist = F.softmax(self.norm(torch.matmul(theta, self.beta)), dim=1)
        self.topic_word_matrix = self.beta
        return word_dist


class ProdLDA(BaseVAE):
    def __init__(self, vocab_size, topic_size, hidden_dims=(100, 100), dropout=0.2,
                 prior_mean=0.0, prior_variance=None, learn_priors=False
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.encoder = InferenceNet(vocab_size, topic_size, hidden_dims=hidden_dims, activation='softplus',
                                    dropout=dropout)
        self.decoder = ProdLDADecoder(self.topic_size, self.vocab_size)
        self.variational_dist = LogisticNormalDistribution(hidden_dims[-1], topic_size)

        self.prior_mean = torch.tensor(
            [prior_mean] * topic_size)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if prior_variance is None:
            prior_variance = 1. - (1. / self.topic_size)
        self.prior_variance = torch.tensor(
            [prior_variance] * topic_size)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.learn_priors = learn_priors
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

    def compute_loss(self, x=None, recon_x=None, hidden=None, theta=None, posterior=None):
        # KL term
        # prior = Normal(self.prior_mean, self.prior_variance)
        # prior = Normal(torch.zeros(self.topic_size, device=x.device), torch.ones(self.topic_size, device=x.device))

        posterior_mean = posterior.loc
        posterior_log_variance = 2 * torch.log(posterior.scale)

        posterior_variance = torch.exp(0.5 * posterior_log_variance)
        var_division = torch.sum(posterior_variance / self.prior_variance, dim=1)
        # diff means term
        diff_means = self.prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / self.prior_variance, dim=1)

        # logvar det division term
        logvar_det_division = \
            self.prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.topic_size + logvar_det_division)
        # KL = torch.distributions.kl.kl_divergence(posterior, prior).sum()
        # Reconstruction term
        RL = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1)
        loss = RL + KL
        return loss.sum(), KL, RL


class NVDM(BaseVAE):
    def __init__(self, vocab_size, topic_size, hidden_dims=(100, 100), dropout=0.2,
                 prior_mean=0.0, prior_variance=None, learn_priors=False):
        super().__init__()
        self.encoder = InferenceNet(vocab_size, topic_size, activation='softplus')
        self.decoder = ProdLDADecoder(self.topic_size, self.vocab_size)


class ProdLDAGenNet(nn.Module):
    def __init__(self, latent_size, output_size, dropout=0.2):
        super().__init__()
        self.beta = torch.Tensor(latent_size, output_size)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(num_features=output_size, affine=False)
        self.dropout_theta = nn.Dropout(p=dropout)

    def forward(self, theta):
        theta = self.dropout_theta(theta)
        # ProdLDA
        recon_x = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        topic_word = self.beta
        return recon_x, topic_word


class ProdLDAFCGenNet(nn.Module):
    def __init__(self, topic_size, vocab_size, dropout=0.2):
        super().__init__()

        self.beta = nn.Linear(topic_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(vocab_size)
        self.batch_norm.weight.requires_grad = False
        # self.beta = None
        # self.batch_norm.register_parameter('weight', None)

    def forward(self, theta):
        theta = self.dropout(theta)
        recon_x = self.batch_norm(self.beta(theta))  # F.softmax(, dim=1)
        return F.softmax(recon_x, dim=1), self.beta.weight.T


class ProdLDAAuthors(nn.Module):
    def __init__(self, vocab_size, topic_size, hidden_dims=(100, 100), dropout=0.2,
                 prior_mean=0.0, prior_variance=None, learn_priors=True
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.encoder = InferenceNet(vocab_size, topic_size, hidden_dims=hidden_dims, activation='softplus')

        self.f_mu = nn.Linear(self.hidden_dims[-1], topic_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(self.topic_size, affine=False)

        self.f_sigma = nn.Linear(self.hidden_dims[-1], self.topic_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(self.topic_size, affine=False)

        self.decoder = ProdLDAGenNet(latent_size=topic_size, output_size=vocab_size)
        #self.decoder = ProdLDAFCGenNet(topic_size, vocab_size)

        """self.beta = torch.Tensor(topic_size, vocab_size)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(num_features=vocab_size, affine=False)
        self.dropout_theta = nn.Dropout(p=dropout)"""
        # self.variational_dist = LogisticNormalDistribution(hidden_dims[-1], topic_size)

        self.prior_mean = torch.tensor(
            [prior_mean] * topic_size)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if prior_variance is None:
            prior_variance = 1. - (1. / self.topic_size)
        self.prior_variance = torch.tensor(
            [prior_variance] * topic_size)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.learn_priors = learn_priors
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

    def reparameterize(self, mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), dist  # eps.mul(std).add_(mu)

    def forward(self, x, *args):
        hidden = self.encoder(x)

        posterior_mu = self.f_mu_batchnorm(self.f_mu(hidden))
        posterior_log_sigma = self.f_sigma_batchnorm(self.f_sigma(hidden))

        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        z, posterior_dist = self.reparameterize(posterior_mu, posterior_log_sigma)
        theta = F.softmax(z, dim=1)
        topic_doc = theta

        # theta = self.dropout_theta(theta)

        # ProdLDA
        # recon_x = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        # topic_words = self.beta
        recon_x, topic_words = self.decoder(theta)
        # word_dist = F.softmax(recon_x, dim=1)

        # posterior_dist = SimpleNamespace(loc=posterior_mu, scale=posterior_sigma, log_scale=posterior_log_sigma)
        return recon_x, hidden, theta, posterior_dist, topic_words

    def compute_loss(self, x=None, recon_x=None,
                     posterior=None, **args):
        def _kl_log_normal(posterior, prior):
            pass

        def _kl_normal_normal(p, q):
            var_ratio = (p.scale / q.scale).pow(2)
            t1 = ((p.loc - q.loc) / q.scale).pow(2)
            return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

        posterior_variance = posterior.scale
        posterior_mean = posterior.loc
        posterior_log_variance = 2 * torch.log(posterior.scale)
        latent_loss = 0.5 * (torch.sum(torch.div(posterior_variance, self.prior_variance), 1) +
                             torch.sum(
                                 (self.prior_mean - posterior_mean).pow(2) / self.prior_variance, dim=1)
                             - self.topic_size +
                             torch.sum(torch.log(self.prior_variance)) - torch.sum(posterior_log_variance, dim=1)
                             )
        prior = Normal(self.prior_mean, self.prior_variance)
        # latent_loss = torch.distributions.kl_divergence(posterior, prior).sum()
        # latent_loss = torch.distributions.kl_divergence(LogNormal(posterior.mean, posterior.scale),
        #                                                LogNormal(self.prior_mean, self.prior_variance)).sum()
        # Reconstruction term
        recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1)
        loss = latent_loss + recon_loss
        return latent_loss, recon_loss

    def get_beta(self, **args):
        return self.decoder.beta
