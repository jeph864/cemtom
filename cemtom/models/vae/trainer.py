import datetime
import json
import os
from collections import OrderedDict, defaultdict

import mlflow
import numpy as np
import torch

from .vae import BaseVAE


class Trainer:
    def __init__(self, model: BaseVAE, train_loader, val_loader, optimizer, device,
                 model_path=None, hyperparameters=None
                 ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        # Initialize placeholders for the best distributions
        self.best_doc_topic_dist = None
        self.best_topic_word_dist = None
        self.best_val_loss = float('inf')
        self.best_loss_train = None

        self.hyperparameters = hyperparameters
        if self.hyperparameters is None:
            self.hyperparameters = {'name': self.model.__class__.__name__.lower()}
        assert isinstance(self.hyperparameters, dict), "Hyperparameters should be a dictionary"

        self.save_dir = model_path
        if self.save_dir is None:
            self.save_dir = os.path.join('models', self.model.__class__.__name__.lower())
        self.save_dir = os.path.join(self.save_dir, self._format_file())
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.early_stopping = EarlyStopping(patience=5, verbose=False,
                                            path=os.path.join(self.save_dir, f'checkpoint.pt'))

        self.n_epoch = None
        self.process_logs = []

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        samples_processed = 0
        doc_topics = []
        topic_words = []
        for batch in self.train_loader:
            x = batch['X'].to(self.device)
            x_embeds = None
            if 'X_emb' in batch:
                x_embeds = batch['X_emb'].to(self.device)
            self.model.zero_grad()
            recon_x, hidden, topic_document, posterior, topic_words = self.model(x, x_embeds)

            kl, nll = self.model.compute_loss(x=x, recon_x=recon_x,
                                              posterior=posterior)  # Assuming reconstruction loss for VAE

            loss = kl + nll
            loss = loss.sum()
            loss.backward()
            # for name, param in self.model.decoder.named_parameters():
            self.optimizer.step()
            samples_processed += x.size()[0]
            running_loss += loss.item()
            # running_kl += kl.item()
            total_kl_loss += kl.sum().item()
            total_recon_loss += nll.sum().item()

            doc_topics.append(topic_document.cpu())
        epoch_loss = running_loss / samples_processed
        avg_recon_loss = total_recon_loss / samples_processed
        avg_kl_loss = total_kl_loss / samples_processed
        self.document_topic_matrix = torch.cat(doc_topics, dim=0)
        self.topic_word_matrix = topic_words  # self.model.get_beta(normalized=False)

        return epoch_loss, avg_kl_loss, avg_recon_loss, samples_processed

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        samples_processed = 0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['X'].to(self.device)
                x_embeds = None
                if 'X_emb' in batch:
                    x_embeds = batch['X_emb'].to(self.device)
                recon_x, hidden, z, posterior, _ = self.model(x, x_embeds)
                kl, nll = self.model.compute_loss(x=x, recon_x=recon_x,
                                                  posterior=posterior)  # Assuming reconstruction loss for VAE
                loss = kl + nll
                loss = loss.sum()
                samples_processed += x.size(0)
                running_loss += loss.item()  # * x.size(0)
                total_kl_loss += kl.sum().item()  # * x.size(0)
                total_recon_loss += nll.sum().item()  # * x.size(0)
        epoch_loss = running_loss / samples_processed
        avg_recon_loss = total_recon_loss / samples_processed
        avg_kl_loss = total_kl_loss / samples_processed

        # Check if this is the best model so far
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            # Assuming model has methods to get distributions; modify as needed
            self.best_doc_topic_dist = self.document_topic_matrix
            self.best_topic_word_dist = self.topic_word_matrix  # self.model.get_beta(normalized=False)

        return epoch_loss, avg_kl_loss, avg_recon_loss, samples_processed

    def fit(self, num_epochs, save_dir=None):
        # Log hyperparameters at the beginnin

        for epoch in range(num_epochs):
            self.n_epoch = epoch
            trn_start = datetime.datetime.now()
            train_loss, trn_kl_loss, trn_recon_loss, trn_sp = self.train_epoch()
            trn_end = datetime.datetime.now()

            val_start = datetime.datetime.now()
            val_loss, val_kl_loss, val_recon_loss, val_sp = self.evaluate()
            val_end = datetime.datetime.now()
            self.log_dict(OrderedDict({
                'epoch': epoch + 1,
                'trn_samples': trn_sp,
                'val_samples': val_sp,
                'train/loss': train_loss,
                'train/kl': trn_kl_loss,
                'train/recon': trn_recon_loss,
                'train/time': trn_end - trn_start,
                'val/loss': val_loss,
                'val/kl': val_kl_loss,
                'val/recon': val_recon_loss,
                'val/time': val_end - val_start
            }), epoch + 1)
            if np.isnan(val_loss) or np.isnan(train_loss):
                break
            else:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    if save_dir is not None:
                        self.save(save_dir)
                    break
        topics = self.get_topics(self.train_loader.dataset.idx2token)
        return topics

    def log_dict(self, log_data, epoch):

        log_message = ' | '.join(
            [f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}' for key, value in
             log_data.items()])
        print(log_message)
        if 'train/time' in log_data:
            log_data['train/time'] = log_data['train/time'].total_seconds()
        if 'val/time' in log_data:
            log_data['val/time'] = log_data['val/time'].total_seconds()
        log_data.pop('epoch')

        self.process_logs.append(log_data)

    def predict(self, data_loader, samples=20):
        assert samples > 0
        self.model.eval()
        predictions = []
        final_preds = []
        for sample_idx in range(samples):
            with torch.no_grad():
                all_preds = []
                for inputs in data_loader:
                    inputs = inputs['X'].to(self.device)
                    x_embeds = None
                    if 'X_emb' in inputs:
                        x_embeds = inputs['X_emb'].to(self.device)
                    self.model.zero_grad()
                    _, _, topic_document, _, _ = self.model(inputs, x_embeds)
                    all_preds.extend(topic_document.cpu().numpy().tolist())
                final_preds.append(np.array(all_preds))
        return np.sum(final_preds, axis=0) / samples

    def save(self, model_dir=None):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        model_dir = self.save_dir
        filename = "epoch_{}".format(self.n_epoch) + '.pth'

        fileloc = os.path.join(model_dir, filename)
        with open(fileloc, 'wb') as file:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'dcue_dict': self.__dict__
            }, file)
        print(f"Saved model at {fileloc}")

    def load(self, path):
        with open(path, 'rb') as file:
            checkpoint = torch.load(file)
        if 'dcue_dict' in checkpoint:
            for (k, v) in checkpoint['dcue_dict'].items():
                setattr(self, k, v)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_topics(self, idx2token, k=10):
        best_component = self.best_topic_word_dist.data
        topic_size = best_component.shape[0]
        k = 10
        topics_list = []
        topics = defaultdict(list)
        for i in range(topic_size):
            _, idxs = torch.topk(best_component[i], k)
            component_words = [idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = component_words
            topics_list.append(component_words)
        return topics_list

    def save_topics(self, topics, evaluation):
        fileloc = os.path.join(self.save_dir, f"{self.n_epoch}_topics.json")
        topics = [" ".join(topic) for topic in topics]
        with open(fileloc, 'w') as file:
            json.dump({'topics': topics, 'evaluation': evaluation}, file)

    def _format_file(self):
        model_dir = "{}_topics_{}_ac_{}_do_{}_lr_{}_mo_{}".format(
            self.model.__class__.__name__.lower(),
            self.hyperparameters['topic_size'],
            self.hyperparameters['activation'],
            self.hyperparameters['dropout'],
            self.hyperparameters['lr'],
            self.hyperparameters['momentum'],
        )
        if 'embedding_size' in self.hyperparameters:
            model_dir = model_dir + f'_emb_{self.hyperparameters["embedding_size"]}'
        return model_dir


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
        with open(self.path, 'wb') as out:
            torch.save({
                'model_state_dict': model.state_dict()
            }, out)
        self.val_loss_min = val_loss
