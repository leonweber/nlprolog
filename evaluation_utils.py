import re

from model import sim
import json
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def s(p, i, m):
    t1 = ["", "", ""]
    t1[0] = ' '.join(p['query'].split()[1:]).lower()
    t1[1] = p['query'].split()[0].replace('_', ' ').lower()
    t1[2] = p['answer'].lower()

    t2 = [e.lower() for e in p['supports'][i]]

    s0 = sim(m.embed_sentences(t1[0:1]), m.embed_sentences(t2[0:1]))
    s1 = sim(m.embed_sentences(t1[1:2]), m.embed_sentences(t2[1:2]))
    s2 = sim(m.embed_sentences(t1[2:3]), m.embed_sentences(t2[2:3]))

    print("t1:", t1)
    print("t2:", t2)
    print("s0:", s0, "s1:", s1, "s2:", s2)
    print("Min: ", min(s0, s1, s2))
    print("Prod: ", s0 * s1 * s2)


def search(prob, s):
    result = []
    for i, (e1, p, e2) in enumerate(prob['supports']):
        if s in e1.lower() or s in p.lower() or s in e2.lower():
            result.append((i, (e1, p, e2)))
    return result

def search_text(prob, s):
    result = []
    for i, doc in enumerate(prob['supports']):
        if s in doc:
            result.append((i, doc))
    return result




def load(path, fname):
    with open(path / 'dev_scores.json') as f:
        scores = json.load(f)

    with open(path / 'dev_depths.json') as f:
        depths = json.load(f)

    with open(path / 'dev_rules.json') as f:
        rules = json.load(f)

    with open(fname) as f:
        data = json.load(f)

    candidates = [d['candidates'] for d in data]
    answers = [d['answer'] for d in data]
    queries = [d['query'] for d in data]

    multihop = []
    for d in data:
        mh = 0
        for ann in d['annotations']:
            if ann[1] == 'multiple':
                mh += 1
        multihop.append(mh)

    pred_indices = [np.argmax(s) for s in scores]
    true_indices = []

    if answers[0] not in {True, False}:
        for answer, cs in zip(answers, candidates):
            for i, candidate in enumerate(cs):
                if answer == candidate:
                    true_indices.append(i)
                    break
        correct = np.array(pred_indices) == np.array(true_indices)[:len(pred_indices)]
    else:
        correct = []
        for d, score in zip(data, scores):
            #         score = score[0]
            if d['answer']:
                correct.append(score >= 0.5)
            else:
                correct.append(score < 0.5)
        true_indices = [0] * len(data)

    answers = np.array(answers)
    scores = np.array(scores)
    depths = np.array(depths)
    correct = np.array(correct)
    multihop = np.array(multihop)

    return pd.DataFrame({'correct': correct, 'depth': depths, 'multihop': multihop})


def evaluate(data, preds):
    res = []
    for prob, pred in zip(data, preds):
        res.append(prob['answer'] == pred)
    return np.array(res)


def evaluate_s2v(data, preds, m):
    res = []
    for prob, pred in zip(data, preds):
        cand_embs = m.embed_sentences(prob['candidates'])
        pred_emb = m.embed_sentences([pred])
        sims = cosine_similarity(cand_embs, pred_emb)
        pred = prob['candidates'][np.argmax(sims, axis=0)[0]]
        res.append(prob['answer'] == pred)
    return np.array(res)


def print_scores(data, nlprolog, neural, m):
    neural_preds = [neural.get(p['id'], '') for p in data if p['id']]
    neural_result = evaluate(data, neural_preds)
    neural_s2v_result = evaluate_s2v(data, neural_preds, m)
    rule_usage = nlprolog['depth'] > 0
    ensemble = np.where(rule_usage, nlprolog['correct'], neural_s2v_result)
    multihop = nlprolog['multihop'] > 2

    print(f"FULL\n"
          f"_____\n"
          f"Neural: {neural_result.mean()}\n"
          f"Neural (s2v): {neural_s2v_result.mean()}\n"
          f"NLProlog: {nlprolog['correct'].mean()}\n"
          f"Ensemble: {ensemble.mean()}\n")

    print(f"MULTIHOP\n"
          f"________\n"
          f"Neural: {neural_result[multihop].mean()}\n"
          f"Neural (s2v): {neural_s2v_result[multihop].mean()}\n"
          f"NLProlog: {nlprolog['correct'][multihop].mean()}\n"
          f"Ensemble: {ensemble[multihop].mean()}\n")


def interpret_rules(model, vocab, embeddings, rules):
    sym_embs = np.array(model['symbol_embedding'])
    sent_embs = []
    for sent in vocab:
        sent_embs.append(embeddings.embed_sentence(sent))
    vocab += [sym for sym in model['symbols'] if not sym.startswith("DUMMY")]
    sent_embs += [sym_embs[i] for sym, i in model['symbols'].items() if not sym.startswith("DUMMY")]
    sym_sent_sims = (cosine_similarity(sym_embs, sent_embs) + 1)/2

    new_rules = []
    for rule in rules:
        score = 1.0
        syms = re.findall(r'DUMMY_SYMBOL_\d', rule)
        for sym in syms:
            idx = model['symbols'][sym]
            sent_idx = sym_sent_sims[idx].argmax()
            sent = vocab[sent_idx]
            rule = rule.replace(sym, sent)
            score *= sym_sent_sims[idx, sent_idx]
        new_rules.append((rule, score))
    return new_rules

# import sent2vec
# from model import sim
#
#
# m = sent2vec.Sent2vecModel()
# m.load_model('/home/leon/data/embeddings/sent2vec_wiki_unigrams.bin')
#
# with open('evaluation/publisherhop_random/model.json') as f:
#     model = json.load(f)
# with open('data/publisherhop_minie/random_rules.txt') as f:
#     rules = f.read().strip().split('\n')
# with open('data/publisherhop_minie/vocab.txt') as f:
#     vocab = f.read().strip().split('\n')
#
# new_rules = interpret_rules(model, vocab, m, rules)
# print(sorted(new_rules, key=lambda x: x[1])[::-1])
