import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, LogNormal, LogisticNormal, kl_divergence


class H2Normal(nn.Module):
    def __init__(self, hidden_dim, latent_dim, prior_mean=0.0, prior_variance=None, learn_priors=True,
                 batch_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.f_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.f_mu_batchnorm = nn.BatchNorm1d(self.latent_dim, affine=False)
        self.f_sigma = nn.Linear(self.hidden_dim, self.latent_dim)
        self.f_sigma_batchnorm = nn.BatchNorm1d(self.latent_dim, affine=False)

        # Move Later ???
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

    def kl_divergence_analytic(self, posterior):
        posterior_variance = posterior.scale
        posterior_mean = posterior.loc
        posterior_log_variance = 2 * torch.log(posterior.scale)

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
        latent_loss = 0.5 * (var_division + diff_term - self.latent_dim + logvar_det_division)
        latent_loss = torch.distributions.kl_divergence(posterior, Normal(self.prior_mean, self.prior_variance)).sum(-1)
        return latent_loss

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
            # eps = torch.randn_like(std)
            z = dist.rsample()
            # z = eps.mul(std).add_(mu)
        else:
            z = dist.mean
            # z = mu
        return F.softmax(z, dim=1), dist


def calc_epsilon(p, alpha):
    sqrt_alpha = torch.sqrt(9 * alpha - 3)
    powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
    return sqrt_alpha * (powza - 1)


def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = u.shape[0]
    K = alpha.shape[1]
    r = torch.arange(B, device=alpha.device)
    rm = torch.reshape(r, (-1, 1, 1)).float()
    alpha_vec = torch.tile(alpha, (B, 1)).reshape((B, -1, K)) + rm
    u_pow = torch.pow(u, 1. / alpha_vec) + 1e-10
    return torch.prod(u_pow, axis=0) * gamma_h(epsilon, alpha + B)


def gamma_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return b * (v ** 3)


def rsvi(alpha):
    B = 10
    gam = torch.distributions.Gamma(alpha + B, 1).sample().to(alpha.device)
    eps = calc_epsilon(gam, alpha + B).detach().to(alpha.device)
    u = torch.rand((B, alpha.shape[0], alpha.shape[1]), device=alpha.device)
    doc_vec = gamma_h_boosted(eps, u, alpha)
    # normalize
    gam = doc_vec
    doc_vec = gam / torch.reshape(torch.sum(gam, dim=1), (-1, 1))
    z = doc_vec.reshape(alpha.shape)
    return z


class H2Dirichlet(nn.Module):
    def __init__(self, hidden_dim, latent_dim, prior_concentration=1.0, learn_priors=False, samples="vanilla"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learn_priors = learn_priors
        self.samples = samples
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn = nn.BatchNorm1d(self.latent_dim, affine=False)

        self.prior_concentration = torch.tensor([prior_concentration] * self.latent_dim)
        if torch.cuda.is_available():
            self.prior_concentration = self.prior_concentration.cuda()
        if learn_priors:
            self.prior_concentration = nn.Parameter(self.prior_concentration)

    def forward(self, hidden):
        alphas = F.softplus(self.bn(self.fc(hidden)))
        min_value = torch.tensor(0.00001)
        if torch.cuda.is_available():
            min_value = min_value.cuda()
        alphas = torch.max(min_value, alphas)
        dist = Dirichlet(alphas)
        if self.samples == "vanilla":
            if self.training:
                z = dist.rsample()
            else:
                z = dist.mean
        elif self.samples == "rsvi":
            z = rsvi(alphas)
        return z, dist
        # return F.softmax(z, dim=1), dist

    def kl_divergence_analytic(self, posterior, beta=3.30):
        prior_concentration = None
        if self.learn_priors:
            prior_concentration = F.softplus(self.prior_concentration)
        else:
            prior_concentration = self.prior_concentration * 0.002
        prior = Dirichlet(prior_concentration)
        return beta * torch.distributions.kl_divergence(posterior, prior).mean()
