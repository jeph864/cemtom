import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, LogNormal, LogisticNormal, kl_divergence
from types import SimpleNamespace


class VariationalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def reparameterize(self, *params):
        raise NotImplementedError


class NormalDistribution(VariationalDistribution):
    def reparameterize(self, params):
        mu, logvar = params
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class LogNormalDistribution(VariationalDistribution):
    def __init__(self, hidden_dim, latent_dim):
        super(LogNormalDistribution, self).__init__()
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # Add BatchNorm layers
        self.mean_bn = nn.BatchNorm1d(latent_dim)
        self.logvar_bn = nn.BatchNorm1d(latent_dim)

    def reparameterize(self, hidden_representation):
        mu = self.mean_bn(self.mean(hidden_representation))
        logvar = self.logvar_bn(self.logvar(hidden_representation))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return torch.exp(mu + eps * std), Normal(mu, std)  # Reparameterization trick for LogNormal


class LogisticNormalDistribution(VariationalDistribution):
    def __init__(self, hidden_dim, latent_dim, batch_norm=False):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # Add BatchNorm layers
        self.mean_bn = nn.BatchNorm1d(latent_dim)
        self.logvar_bn = nn.BatchNorm1d(latent_dim)

    def reparameterize(self, hidden_representation):
        mu = self.mean_bn(self.mean(hidden_representation))
        logvar = self.logvar_bn(self.logvar(hidden_representation))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick for underlying normal
        # print(std)
        return F.softmax(z, dim=1), SimpleNamespace(loc=mu, scale=std)  # Map to simplex for Logistic Normal


class DirichletDistribution(VariationalDistribution):
    def __init__(self):
        super(DirichletDistribution, self).__init__()

    def reparameterize(self, alpha):
        # Stick-breaking or other methods can be more complex; this uses a simple gamma-based approach
        gamma_samples = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)  # Reparameterization for Dirichlet
