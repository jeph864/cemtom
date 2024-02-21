from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Dirichlet, kl_divergence


class VAEBase(pl.LightningModule):
    def __init__(self, in_features, num_topics, hidden_dims=(100,), learning_rate=0.001):
        super().__init__()
        self.in_features = in_features
        self.num_topics = num_topics
        self.learning_rate = learning_rate

        # Encoder
        self.encoder = InferenceNet(in_features=self.in_features, out_features=self.num_topics, affine=False)

        # Decoder
        self.decoder = nn.Linear(num_topics, self.in_features)
        self.decoder_norm = lambda x: x  ## by default the BatchNormalization will be  an identity function

        self.save_hyperparameters()

    def encode(self, x):
        encoder_out = F.softplus(self.encoder.out(x))
        mu, sigma = torch.chunk(encoder_out, 2, dim=1)
        return mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        return z, dist

    def decode(self, z):
        return torch.softmax(self.decoder_norm(self.decoder(z)), dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, dist = self.reparameterize(mu, logvar)
        return self.decode(z), dist

    def training_step(self, batch, batch_idx):
        x = batch.float()
        recon_x, dist = self(x)
        recon_loss, kl_loss = self.objective(x, recon_x, dist)
        loss = recon_loss + kl_loss
        self.log_dict({'train/loss': loss,
                       'train/recon': recon_loss,
                       'train/kl': kl_loss},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def objective(self, x, x_recon, dist):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        prior = Normal(torch.zeros(self.num_topics, device=x.device), torch.ones(self.num_topics, device=x.device))
        kl = self.beta * torch.distributions.kl.kl_divergence(dist, prior).mean()
        return recon, kl

    def validation_step(self, batch, batch_idx):
        x = batch.float()
        recon_x, dist = self(x)
        recon_loss, kl_loss = self.objective(x, recon_x, dist)
        loss = recon_loss + kl_loss
        self.log_dict({'val/loss': loss,
                       'val/recon': recon_loss,
                       'val/kl': kl_loss},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def get_topic_words(self, vocab, path):
        # load best model
        model = self.__class__.load_from_checkpoint(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get topics
        topics = model.decoder.weight.detach().cpu().numpy().T
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]
        return topics


class ProdLDA(VAEBase):
    def __init__(self, in_features, num_topics, beta=2.0):
        super().__init__(in_features, num_topics)
        self.beta = beta
        self.decoder_norm = nn.BatchNorm1d(num_features=self.in_features, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.in_features))
        self.decoder_norm.weight.requires_grad = False


class InferenceNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=(100,), activation='relu', dropout=0.25, bn=True,
                 affine=False, *args,
                 **kwargs):
        super(InferenceNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout

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
        self.output_layer = nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_features * 2)

        self.mu = nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_features)
        self.sigma = nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_features)
        # batch normalization
        self.mu_norm = None
        self.sigma_norm = None
        if bn:
            self.mu_norm = nn.BatchNorm1d(num_features=self.out_features, eps=0.001, momentum=0.001, affine=affine)
            self.sigma_norm = nn.BatchNorm1d(num_features=self.out_features, eps=0.001, momentum=0.001, affine=affine)

    def forward(self, x):
        inference_out = self.out(x)
        mu, sigma = torch.chunk(inference_out, 2, dim=1)
        if self.mu_norm is not None:
            mu = self.mu_norm(mu)
            sigma = self.sigma_norm(sigma)
        return mu, sigma

    def out(self, x):
        return self.output_layer(self.dropout(self.hidden_layers(self.activation(self.input(x)))))
