import pdb

import fasttext.util
import fasttext

import os
import time, sys

import nltk
import torch
import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from pytorch_transformers import BertModel as PyBertModel, BertTokenizer as PyBertTokenizer
from transformers import BertTokenizer, BertTokenizerFast, BertModel, GPT2TokenizerFast


class BaseEmbedder:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.word2idx = None

    def embed_word(self, word):
        pass

    def embed_doc(self, doc):
        pass

    def embed(self, doc):
        pass


class BaseTokenEmbedder(BaseEmbedder):
    def __init__(self, model_name=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings_ = None
        self.tokens_ = None
        self.w2idx = None


class BertTokenEmbedder(BaseTokenEmbedder):
    def __init__(self, model_name=None, layer=-1, max_seq_length=512, batch_size=1):
        super().__init__()
        self.layer = layer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model_name = model_name if model_name is not None else 'bert-base-uncased'
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            print("Using GPU(s)")
            self.model = self.model.to('cuda')
        self.embedding_vocab = None

    def embed(self, docs):
        tokens = []
        words_ids = []
        for document in tqdm(docs, desc="Tokenizing and mapping"):
            token_ids, word_idx = self._tokenize_and_map(document)
            tokens.append(token_ids)
            words_ids.append(word_idx)
        self.word2idx = words_ids
        self.tokens_ = tokens
        org_word_mapping = []
        for doc, word_ids in tqdm(zip(self.tokens_, self.word2idx), total=len(tokens), desc="Reconstructing  original "
                                                                                            "words"):
            word_mapping = self._reconstruct_doc(doc, word_ids)
            org_word_mapping.append(word_mapping)
        self.embedding_vocab = [word for d in org_word_mapping for word in d]
        word_embeddings = []
        for token_ids, word_ids in tqdm(zip(tokens, words_ids), total=len(tokens), desc="embedding documents"):
            embedding = self._process_block(token_ids, word_ids)
            word_embeddings.append(embedding)
        embedding_shape = word_embeddings[0][0].size()[0]
        # Concatenate all embeddings into a single tensor
        # Flatten the list of lists and then concatenate
        flat_list_of_tensors = [emb for sublist in word_embeddings for emb in sublist]
        self.embeddings_ = torch.cat(
            flat_list_of_tensors, dim=0).cpu().numpy().reshape(-1, embedding_shape)
        # del flat_list_of_tensors

    def _reconstruct_doc(self, tokens, word_ids):
        doc = []
        for subword, token_id in zip(tokens, word_ids):
            if token_id is None:  # special characters
                continue
            if subword.startswith('##'):
                doc[-1] += subword.replace('##', '')
            else:
                doc.append(subword)
        return doc

    def _tokenize_and_map(self, document: str):
        """
        Tokenizes a document and maps tokens to their original words.
        """
        encoded = self.tokenizer(document, return_offsets_mapping=True, padding=True, truncation=True,
                                 max_length=self.max_seq_length)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded.input_ids)
        offsets = encoded.offset_mapping
        word_ids = [offset[0] for offset in offsets]
        return tokens, encoded.word_ids()

    def _process_block(self, block_tokens, block_word_ids):
        """
        Processes a block of tokens through BERT and averages subword embeddings to reconstitute word embeddings.
        """
        inputs = self.tokenizer(block_tokens, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_seq_length,
                                is_split_into_words=True)
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Assuming batch size of 1 for simplicity

        # Average subword embeddings for each word
        word_embeddings = []
        current_word_id = None
        current_word_embeddings = []
        for token_id, embedding in zip(block_word_ids, embeddings):
            if token_id is None:
                continue
            if token_id != current_word_id:
                if current_word_embeddings:
                    word_embeddings.append(torch.mean(torch.stack(current_word_embeddings), dim=0))
                current_word_embeddings = [embedding]
                current_word_id = token_id
            else:
                current_word_embeddings.append(embedding)
        if current_word_embeddings:  # Handle last word
            word_embeddings.append(torch.mean(torch.stack(current_word_embeddings), dim=0))

        return word_embeddings

    def _average_subword_embeddings(self, tokenized_text, embeddings):
        word_embeddings = []
        current_word = []
        for token, embedding in zip(tokenized_text, embeddings):
            if token.startswith("##"):
                current_word.append(embedding)
            else:
                if current_word:
                    word_embeddings.append(torch.mean(torch.stack(current_word), dim=0))
                current_word = [embedding]
        # Handle last word
        if current_word:
            word_embeddings.append(torch.mean(torch.stack(current_word), dim=0))
        return word_embeddings

    def _divide_tokens_into_blocks(self, tokens, max_length):
        blocks = []
        current_block = []
        current_length = 0

        for token in tokens:
            # BERT uses "##" to denote subtokens belonging to the same word
            is_subtoken = token.startswith("##")
            token_length = 1 if not is_subtoken else 0  # Subtokens don't add to length if part of a continuing word

            if current_length + token_length > max_length:
                # Finish the current block and start a new one
                blocks.append(current_block)
                current_block = [token] if not is_subtoken else []  # Handle edge case where subtoken starts a new block
                current_length = token_length
            else:
                current_block.append(token)
                current_length += token_length

        # Add the last block if it's not empty
        if current_block:
            blocks.append(current_block)

        return blocks

    def _process_document(self, document: str):
        """
        Processes a document from tokenization through embedding generation and reconstitutes word embeddings.
        """
        tokens, word_ids = self._tokenize_and_map(document)
        # Here, you would split into blocks if needed and adjust for very long documents
        # For simplicity, assuming the whole document is processed as one block
        embeddings = self._process_block(tokens, word_ids)
        return embeddings


class BaseWordEmbedder:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.__fit_status = False

    def is_fit(self):
        return self.__fit_status

    def set_fit(self, status=False):
        self.__fit_status = status

    def embed(self, words):
        raise NotImplemented("Subclass should implement the method")


class BertVocabEmbedder(BaseWordEmbedder):
    def __init__(self, device=-1, valid_vocab=-1, pretrained_weights='bert-base-uncased', aggy_by="firstword",
                 embeddings=None,
                 vocab=None,
                 nr_layer=12):
        super().__init__()
        self.device = torch.device("cuda:{}".format(device) if (device is not None and int(device) >= 0) else "cpu")
        self.sentence_tokenizer = nltk.load('tokenizers/punkt/english.pickle')
        self.pretrained_weights = pretrained_weights
        self.tokenizer = PyBertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = PyBertModel.from_pretrained(self.pretrained_weights, output_hidden_states=True).to(self.device)
        self.w2vb, self.w2vc = {}, {}  # embeds sum and counts
        self.compounds = set()
        self.agg_by = aggy_by
        self.nr_layer = nr_layer
        self.use_full_vocab = False

        if valid_vocab == -1:
            self.use_full_vocab = True
        self.valid_vocab = valid_vocab

        # public
        self.embeddings = embeddings
        self.vocab = vocab
        if self.embeddings is not None:
            self.is_fit = True
        self.word2idx = None
        if self.vocab is not None:
            self.word2idx = {word: index for index, word in enumerate(self.vocab)}

    def _add_word(self, compound_word, compound_ixs, embeds):

        word = "".join(compound_word).lower()
        if self.agg_by == "firstword":
            w = compound_ixs[0]
            emb = embeds[w]
        elif self.agg_by == "average":
            total_emb = 0
            for w in compound_ixs:
                total_emb += embeds[w]
            emb = total_emb / len(compound_ixs)

        emb = emb.cpu().detach().numpy()

        if self.use_full_vocab:
            pass
        else:
            if word not in self.valid_vocab:
                return

        if len(compound_ixs) > 1:
            self.compounds.add(word)

        if word in self.w2vb:
            self.w2vb[word] += emb
            self.w2vc[word] += 1
        else:
            self.w2vb[word] = emb
            self.w2vc[word] = 1

    def embed(self, documents=[], dataname="unkn", save_filename=""):
        if len(save_filename):
            save_filename = f"{dataname}-bert-layer{self.nr_layer}-{self.agg_by}.txt"
        start = time.time()
        with torch.no_grad():
            for i, doc in enumerate(documents):
                if i % (int(len(documents) / 100)) == 0:
                    elapsed_time = np.round(time.time() - start, 1)
                    print(f"{i + 1}/{len(documents)} done, elapsed(s): {elapsed_time}")
                    sys.stdout.flush()
                sentences = self.sentence_tokenizer.tokenize(doc)
                for sentence in sentences:
                    words = self.tokenizer.tokenize(sentence)
                    if len(words) > 0:
                        new_sentences, fragment, subwords_count, current_length = [""], "", 0, 0
                        for w in words:
                            subwords_count += 1
                            if w.startswith("##"):
                                fragment += w.replace("##", "")
                                current_length += 1
                            else:
                                if subwords_count > 500:
                                    new_sentences.append("")
                                new_sentences[-1] += " " + fragment
                                fragment = w
                                current_length = 1
                        new_sentences[-1] += " " + fragment
                        new_sentences = [s[1:] for s in new_sentences]
                        if not words == self.tokenizer.tokenize(" ".join(new_sentences)):
                            pdb.set_trace()
                    else:
                        new_sentences = [sentence]
                    for sent in new_sentences:
                        if len(new_sentences) > 1:
                            words = self.tokenizer.tokenize(sent)
                        input_ids = torch.tensor([words]).to(self.device)
                        embedding = self.model(input_ids)[-2:][1][self.nr_layer][0]
                        compound_word, compound_idx = [], []
                        full_word = ""
                        for w, word in enumerate(words):
                            if word.startswith('##'):
                                compound_word.append(word.replace('##', ''))
                                compound_idx.append(w)
                            else:
                                if w != 0:
                                    self._add_word(compound_word, compound_idx, embedding)
                                compound_word = [word]
                                compound_idx = [w]
                            if w == len(words) - 1:
                                self._add_word(compound_word, compound_idx, embedding)
        embeddings = []
        vocab = []
        for word in self.w2vb:
            mean = np.around(self.w2vb[word] / self.w2vc[word], 8)
            vocab.append(word)
            embeddings.append(mean)
        self.vocab = vocab
        self.embeddings = embeddings

        self.__dump_embeddings(save_filename)

    def __dump_embeddings(self, save_path=None):
        if save_path is None:
            pass  # Error handling
        embeddings = []
        for word in self.w2vb:
            mean = np.around(self.w2vb[word] / self.w2vc[word], 8)
            embeddings.append(np.append(word, mean))
        np.savetxt(save_path, np.vstack(embeddings), ftm="%s", delimiter=" ", encoding="utf-8")
        with open('compounds.txt', 'w') as f:
            f.write("\n".join(list(self.compounds)))

    @classmethod
    def load_embeddings(cls, path):
        embeddings, vocab = [], []
        if not os.path.exists(path):
            raise ValueError("Embeddings do  not exist. Generate them first")
        if path is not None:
            for line in open(path):
                embedding = line.split()
                vocab.append(embedding[0])
                embeddings.append(list(map(float, embedding[1:])))

        bert_model = BertVocabEmbedder(embeddings=np.vstack(embeddings), vocab=vocab)
        bert_model.set_fit(True)
        return bert_model


class FasttextEmbedder(BaseWordEmbedder):
    def __init__(self, embedding_model=None, path=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.model_path = path
        self.model = None
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model = fasttext.load_model(self.model_path)
        elif self.embedding_model is not None:
            self.model = self.embedding_model
        else:
            raise ValueError("No Model or path given")

    def embed(self, words):
        word_embeddings = []
        for word in words:
            word_embeddings.append(self.model.get_word_vector(word))
        return np.array(word_embeddings)


def get_word_embedding_model(name=None, model=None, path=None):
    if name == "fasttext":
        return FasttextEmbedder(embedding_model=model, path=path)
    elif name == "bert":
        return BertVocabEmbedder.load_embeddings(path)
    else:
        return BaseWordEmbedder()
