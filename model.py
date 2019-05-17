import nltk
import IPython as I
import json
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch
from nltk import word_tokenize
import sent2vec
import operator


np.random.seed(5005)

def get_linear(shape):
    std = 1. / np.sqrt(shape[1])
    return np.random.uniform(-std, std, shape)

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def sim(x, y):
    sims = (cosine_similarity(x, y) + 1.0)/2.0
    return sims

class Identity(nn.Module):
    def forward(self, x):
        return x


class Projection(nn.Module):
    def __init__(self, n_units):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(n_units, n_units)

    def forward(self, x):
        x = self.proj(x)
        x = F.relu(x)

        return self.dropout(x)


class Sent2Vec(nn.Module):


    def __init__(self, env, config):
        super().__init__()
        self.entity_vocab = None
        self.config = config
        self.symbol_vocab = None
        self.predicate_vocab = None
        self.env = env
        self.sent2vec = sent2vec.Sent2vecModel()
        self.sent2vec.load_model(config['embeddings'])

        self.train_symbol = config['train_symbol']
        self.train_predicate = config['train_predicate']
        self.train_entity = config['train_entity']

        self.symbol_embedding = None
        self.predicate_embedding = None
        self.entity_embedding = None

        self.proj_sym = Projection(config['pred_dim'])
        self.proj_ent = Projection(config['ent_dim'])
        self.proj_pred = Projection(config['pred_dim'])
        self.lambda_ = 1.0


        self.init_embeddings()

    def get_embedding(self, symbol):
        if symbol.startswith('s'):
            symbol = symbol[1:]
            if self.train_symbol and symbol in self.symbol_vocab:
                return self.proj_sym(self.symbol_embedding[self.symbol_vocab[symbol]])
            else:
                return self.proj_sym(torch.tensor(self.sent2vec.embed_sentence(symbol))[0])
        elif symbol.startswith('p'):
            symbol = symbol[1:]
            if self.train_predicate and symbol in self.predicate_vocab:
                return self.proj_pred(self.predicate_embedding[self.predicate_vocab[symbol]])
            else:
                return self.proj_pred(torch.tensor(self.sent2vec.embed_sentence(symbol))[0])

        elif symbol.startswith('e'):
            symbol = symbol[1:]
            if self.train_entity and symbol in self.entity_vocab:
                return self.proj_ent(self.entity_embedding[self.entity_vocab[symbol]])
            else:
                return self.proj_ent(torch.tensor(self.sent2vec.embed_sentence(symbol))[0])

        else:
            print(symbol)
            raise ValueError


    def recompute_score_with_grads(self, unifications, entity_aggregation=operator.mul, predicate_aggregation=operator.mul):
        entity_score = 1.0
        predicate_score = 1.0
        for unification in unifications:
            s1, s2 = unification.split('<>')
            symbol_type = (s1[0], s2[0])
            v1 = self.get_embedding(s1).unsqueeze(0)
            v2 = self.get_embedding(s2).unsqueeze(0)
            s = sim(v1, v2)

            if symbol_type == ('s', 's'):
                s = self.lambda_ * s + (1-self.lambda_)

            if s1.startswith('e'):
                entity_score = entity_aggregation(entity_score, s)
            else:
                predicate_score = predicate_aggregation(predicate_score, s)
        return entity_score * predicate_score

    def init_embeddings(self):
        self.predicate_vocab = {}
        self.symbol_vocab = {}
        self.entity_vocab = {}

        if self.train_predicate:
            for predicate in self.env.predicate_vocab:
                self.predicate_vocab[predicate] = len(self.predicate_vocab)
                if self.predicate_embedding is None:
                    self.predicate_embedding = self.sent2vec.embed_sentences([predicate.replace('_', ' ')])
                else:
                    self.predicate_embedding = np.concatenate((self.predicate_embedding, self.sent2vec.embed_sentences([predicate.replace('_', ' ')])))
            self.predicate_embedding = nn.Parameter(torch.Tensor(self.predicate_embedding))

        if self.train_symbol:
            for symbol in self.env.symbol_vocab:
                self.symbol_vocab[symbol] = len(self.symbol_vocab)
                if self.config["semantic_query_init"]:
                        symbol_embedding = self.sent2vec.embed_sentences([" ".join(symbol.split('_'))])
                else:
                    symbol_embedding = get_linear((1, self.config["pred_dim"]))

                if self.symbol_embedding is None:
                    self.symbol_embedding = symbol_embedding
                else:
                    self.symbol_embedding = np.concatenate((self.symbol_embedding, symbol_embedding))

            self.symbol_embedding = nn.Parameter(torch.Tensor(self.symbol_embedding))

        if self.train_entity:
            for entity in self.env.entity_vocab:
                self.entity_vocab[entity] = len(self.entity_vocab)
                entity_embedding = self.sent2vec.embed_sentences([entity])
                if entity_embedding.sum() == 0:
                    entity_embedding = get_linear((1, self.config["ent_dim"]))

                if self.entity_embedding is None:
                    self.entity_embedding = entity_embedding
                else:
                    self.entity_embedding = np.concatenate((self.entity_embedding, entity_embedding))

            self.entity_embedding = nn.Parameter(torch.Tensor(self.entity_embedding))


    def get_sims(self, obs):
        with torch.no_grad():
            if self.train_predicate:
                predicate_indices = torch.LongTensor([self.predicate_vocab[p] for p in obs['predicates']])
                predicate_embeddings = self.predicate_embedding[predicate_indices]
            else:
                predicate_embeddings = torch.tensor(self.sent2vec.embed_sentences(obs['predicates']))

            if self.train_symbol:
                symbols = np.array(obs['symbols'])
                symbol_indices = torch.LongTensor([self.symbol_vocab.get(s, -1) for s in obs['symbols']])
                no_emb_mask = symbol_indices == -1
                symbol_embeddings = self.symbol_embedding[symbol_indices]
                symbol_embeddings[no_emb_mask] = torch.tensor(self.sent2vec.embed_sentences(symbols))[no_emb_mask]
            else:
                symbol_embeddings = torch.tensor(self.sent2vec.embed_sentences(obs['symbols']))

            if self.train_entity:
                entity_indices = torch.LongTensor([self.entity_vocab.get(s, -1) for s in obs['entities']])
                entity_embeddings = self.entity_embedding[entity_indices]
                no_emb_mask = entity_indices == -1
                entity_embeddings[no_emb_mask] = torch.tensor(self.sent2vec.embed_sentences(obs['entities']))[no_emb_mask]
            else:
                entity_embeddings = torch.tensor(self.sent2vec.embed_sentences(obs['entities']))

            symbol_embeddings = self.proj_sym(symbol_embeddings)
            entity_embeddings = self.proj_ent(entity_embeddings)
            predicate_embeddings = self.proj_pred(predicate_embeddings)

            sym_pred = sim(symbol_embeddings, predicate_embeddings)
            ent_ent = sim(entity_embeddings, entity_embeddings)
            sym_sym = self.lambda_ * sim(symbol_embeddings, symbol_embeddings) + (1 - self.lambda_)

            return {
                'symbol_predicate_similarity': sym_pred.numpy(),
                'entity_similarity': ent_ent.numpy(),
                'symbol_similarity': sym_sym.numpy()
            }

    def forward(self, obs):
        pass

    def save(self, fname):
        meta = {}
        meta['symbols'] = self.symbol_vocab
        meta['entities'] = self.entity_vocab
        meta['predicates'] = self.predicate_vocab
        meta['config'] = self.config

        with open(fname + '.json', 'w') as f:
            json.dump(meta, f)

        torch.save(self.state_dict(), fname + '.pto')

    def load(self, fname):
        with open(fname + '.json') as f:
            meta = json.load(f)

        self.symbol_vocab = meta['symbols']
        self.entity_vocab = meta['entities']
        self.predicate_vocab = meta['predicates']

        self.load_state_dict(torch.load(fname + '.pto'))
