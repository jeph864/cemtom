from collections import OrderedDict
import numpy as np
import datetime

from cemtom.models.vae.distributions import *


class BaseVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, variational_dist=None):
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
        z, posterior = self.reparameterize(hidden)

        recon_x = self.decoder(z)
        return recon_x, hidden, z, posterior

    def compute_loss(self, x, recon_x, params, theta, posterior):
        raise NotImplementedError

    def get_theta(self, x):
        self.eval()
        # Assuming the last layer of the encoder outputs parameters for theta
        with torch.no_grad():
            params = self.encoder(x)
            z, _ = self.reparameterize(params)
            theta = F.softmax(z, dim=1)
        return theta

    def get_beta(self):
        self.eval()
        # Assuming beta is a parameter of the decoder
        with torch.no_grad():
            if hasattr(self.decoder, 'beta'):
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
    def __init__(self, in_features, out_features, hidden_dims=(100,), activation='relu', dropout=0.25, batch_norm=True,
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

    def forward(self, x):
        hidden = self.hidden_layers(self.activation(self.input(x)))
        hidden = self.dropout(hidden)
        return hidden


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        # self.fc = nn.Linear(in_features, out_features)
        # self.dropout = nn.Dropout(dropout)
        # self.batch_norm = nn.BatchNorm1d(out_features)
        self.beta = None
        self.topic_word_matrix = None

    def forward(self, theta):
        return self.decode(theta)

    def decode(self, theta):
        #beta = F.log_softmax(self.batch_norm(self.fc(theta)), dim=1)
        #return beta
        raise NotImplementedError


class ProdLDADecoder(Decoder):
    def __init__(self, topic_size, vocab_size, dropout=0.2):
        super().__init__(topic_size, vocab_size)
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

        self.encoder = InferenceNet(vocab_size, topic_size, activation='softplus')
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

    def compute_loss(self, x, recon_x, params, theta, posterior):
        posterior_mean, posterior_log_variance = posterior.mean, posterior.logvar
        # KL term
        # var division term
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

        # Reconstruction term
        RL = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1)
        loss = KL + RL
        return loss.sum()


class Trainer:
    def __init__(self, model: BaseVAE, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        # Initialize placeholders for the best distributions
        self.best_doc_topic_dist = None
        self.best_topic_word_dist = None
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        samples_processed = 0
        doc_topics = []
        for batch in self.train_loader:
            x = batch['X'].to(self.device)

            self.model.zero_grad()
            recon_x, hidden, z, posterior = self.model(x)

            loss = self.model.compute_loss(x, recon_x, hidden, z, posterior)  # Assuming reconstruction loss for VAE
            loss.backward()
            self.optimizer.step()
            samples_processed += x.size()[0]
            running_loss += loss.item()
            doc_topics.append(z.cpu())
        epoch_loss = running_loss / samples_processed
        self.model.document_topic_matrix = torch.cat(doc_topics, dim=0)
        return epoch_loss, samples_processed

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        samples_processed = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['X'].to(self.device)
                recon_x, hidden, z, posterior = self.model(x)
                loss = self.model.compute_loss(x, recon_x, hidden, z, posterior)  # Assuming reconstruction loss for VAE
                samples_processed += x.size()[0]
                running_loss += loss.item()
        epoch_loss = running_loss / samples_processed

        # Check if this is the best model so far
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            # Assuming model has methods to get distributions; modify as needed
            self.best_doc_topic_dist = self.model.document_topic_matrix
            self.best_topic_word_dist = self.model.get_beta()

        return epoch_loss, samples_processed

    def fit(self, num_epochs):

        for epoch in range(num_epochs):
            trn_start = datetime.datetime.now()
            train_loss, trn_sp = self.train_epoch()
            trn_end = datetime.datetime.now()
            val_start = datetime.datetime.now()
            val_loss, val_sp = self.evaluate()
            val_end = datetime.datetime.now()
            self.log_dict(OrderedDict({
                'epoch': epoch + 1,
                'trn_samples': trn_sp,
                'val_samples': val_sp,
                'train/loss': train_loss,
                'train/time': trn_end - trn_start,
                'val/loss': val_loss,
                'val/time': val_end - val_start
            }))

    def log_dict(self, log_data):
        log_message = ' | '.join(
            [f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}' for key, value in
             log_data.items()])
        print(log_message)

    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())
        return torch.cat(predictions)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_doc_topic_dist': self.best_doc_topic_dist,
            'best_topic_word_dist': self.best_topic_word_dist
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_doc_topic_dist = checkpoint['best_doc_topic_dist']
        self.best_topic_word_dist = checkpoint['best_topic_word_dist']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
