from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Dirichlet, LogNormal, LogisticNormal, kl_divergence
import numpy as np
import datetime

from .distributions import VariationalDistribution


class BaseVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, hidden2dist=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.variational_dist = variational_dist
        self.document_topic_matrix = None
        self.topic_word_matrix = None

    def reparameterize(self, params):
        return self.variational_dist.reparameterize(params)

    def forward(self, x):
        hidden = self.encoder(x)

        theta, posterior = self.reparameterize(hidden)

        recon_x = self.decoder(theta)
        return recon_x, hidden, theta, posterior

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_theta(self, x):
        self.eval()
        # Assuming the last layer of the encoder outputs parameters for theta
        with torch.no_grad():
            params = self.encoder(x)
            z, _ = self.reparameterize(params)
            theta = F.softmax(z, dim=1)
        return theta

    def get_beta(self, normalized = True):
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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, encoder, decoder, variational_dist):
        model = BaseVAE(encoder, decoder, variational_dist)
        model.load_state_dict(torch.load(path))
        return model


class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class InferenceNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=(100,100), activation='relu', dropout=0.25, batch_norm=True,
                 affine=False, *args,
                 **kwargs):
        super().__init__()
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

    def forward(self, x):
        hidden = self.hidden_layers(self.activation(self.input(x)))
        hidden = self.dropout(hidden)
        return hidden


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc = nn.Linear(in_features, out_features)
        # self.dropout = nn.Dropout(dropout)
        # self.batch_norm = nn.BatchNorm1d(out_features)
        self.beta = None
        self.topic_word_matrix = None

    def forward(self, theta):
        return self.decode(theta)

    def decode(self, theta):
        # beta = F.log_softmax(self.batch_norm(self.fc(theta)), dim=1)
        # return beta
        raise NotImplementedError


class FCDecoder(Decoder):
    def __init__(self, topic_size, vocab_size):
        super().__init__()

        self.fc = nn.Linear(topic_size, vocab_size)
        # self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(vocab_size)
        # self.batch_norm.weight.requires_grad = False

    def forward(self, theta):
        return F.log_softmax(self.batch_norm(self.fc(theta)), dim=1)
