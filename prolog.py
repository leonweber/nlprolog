import logging
import tempfile
import subprocess
import shlex
import numpy as np
from collections import defaultdict
import copy
import os
import json
import utils


LAMBDA_CUT_PRE = 0.5
K = 1000

def luk(a, b):
    return max(0, a + b - 1)

def prod(a, b):
    return a * b 

def query(facts, similarity, goal, entity_tnorm, predicate_tnorm, min_depth, min_bs_size, lambda_cut, max_depth):
    os.makedirs('tmp', exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir='tmp', delete=True, prefix='facts') as facts_file,\
         tempfile.NamedTemporaryFile(mode='w', dir='tmp', delete=True, prefix='sims') as similarity_file:
        facts_file.write(facts)
        facts_file.flush()
        similarity_file.write(similarity)
        similarity_file.flush()
        lambda_cut = "|".join([str(lambda_cut)] * len(goal.split('|')))

        cmd = f'{utils.BASE_PATH}/spyrolog {facts_file.name} {similarity_file.name} {goal} {max_depth} {lambda_cut} {entity_tnorm}|{predicate_tnorm} {min_bs_size}'
        cmd = shlex.split(cmd)
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except (subprocess.TimeoutExpired, ValueError):
            return [[0, 0, b"", b""] for _ in goal.split('|')]
        logging.debug("spyrolog stdout: " + result.stdout.decode())
        logging.debug("spyrolog stderr: " + result.stderr.decode())
        results = []
        try:
            for r in result.stdout.split(b'\n'):
                if len(r) == 0:
                    continue
                split = r.split(b' ')
                if len(split) < 3:
                    results.append( [float(split[0]), int(split[1]), '', ''] )
                else:
                    results.append( [float(split[0]), int(split[1]), b' '.join(split[3:]).decode(), split[2].decode()] )
            return results
        except ValueError:
            raise RuntimeError(result.stderr)



class Program:
    def __init__(self, facts, rules, candidates):
        self.entity_to_id = None
        self.relation_to_id = None
        self.variable_to_id = None
        self.symbol_to_id = None
        self.build_vocabs(facts, rules, candidates)
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self.id_to_variable = {v: k for k, v in self.variable_to_id.items()}
        self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}
        self.rule_unifications = None
          
        self.facts = self.transform_triples(facts)
        self.rules = self.transform_rules(rules)
        self.candidates = self.transform_triples(candidates, symbol=True)
        self.lambda_cut = 0.5

        self.symbol_predicate_similarity = None
        self.entity_similarity = None
        self.symbol_similarity = None
        self.predicate_similarity = None


    def set_similarities(self, symbol_predicate, entity, symbol, predicate=None):
        self.symbol_predicate_similarity = symbol_predicate
        self.entity_similarity = entity
        self.symbol_similarity = symbol
        self.predicate_similarity = predicate

    def build_vocabs(self, facts, rules, candidates):
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.variable_to_id = {}
        self.symbol_to_id = {}

        rule_triples = set()
        for rule in rules:
            rule_triples.add(tuple(rule['consequent'][0]))
            rule_triples.update([tuple(t) for t in rule['antecedents']])
        rule_triples = list(rule_triples)

        for triple in rule_triples :
            for entity in (triple[0], triple[2]):
                if entity.isupper():
                    if entity not in self.variable_to_id:
                        self.variable_to_id[entity] = -len(self.variable_to_id)-1
                elif entity not in self.entity_to_id:
                    self.entity_to_id[entity] = len(self.entity_to_id)

            symbol = triple[1]
            if symbol not in self.symbol_to_id:
                self.symbol_to_id[symbol] = len(self.symbol_to_id)

        for triple in candidates:
            symbol = triple[1]
            if symbol not in self.symbol_to_id:
                self.symbol_to_id[symbol] = len(self.symbol_to_id)
            for entity in (triple[0], triple[2]):
                if entity not in self.entity_to_id:
                    self.entity_to_id[entity] = len(self.entity_to_id)

        for triple in facts:
            for entity in (triple[0], triple[2]):
                if entity not in self.entity_to_id:
                    self.entity_to_id[entity] = len(self.entity_to_id)

            relation = triple[1]
            if relation not in self.relation_to_id:
                self.relation_to_id[relation] = len(self.relation_to_id)


    def transform_triples(self, triples, symbol=False):
        transformed_facts = []
        for e1, r, e2 in triples:
            e1 = self.variable_to_id[e1] if e1.isupper() else self.entity_to_id[e1]
            e2 = self.variable_to_id[e2] if e2.isupper() else self.entity_to_id[e2]
            if symbol:
                transformed_facts.append((e1, self.symbol_to_id[r], e2))
            else:
                transformed_facts.append((e1, self.relation_to_id[r], e2))
        return np.array(transformed_facts)


    def transform_rules(self, rules):
        transformed_rules = []

        for rule in rules:
            transformed_rule = {'consequent': np.array(self.transform_triples([rule['consequent']][0], symbol=True)),
                                'antecedents': np.array(self.transform_triples(rule['antecedents'], symbol=True))}
            transformed_rules.append(transformed_rule)
        return transformed_rules

    def triples_to_prolog(self, triples):
        res = ""

        for triple in triples:
            res += self.triple_to_prolog(triple) + ".\n"

        return res

    def triple_to_prolog(self, triple, symbol=False, invert=False):
        e1, r, e2 = triple
        e1 = self.id_to_variable[e1] if e1 < 0 else f"e{e1}"
        e2 = self.id_to_variable[e2] if e2 < 0 else f"e{e2}"

        if invert:
            e1, e2 = e2, e1

        if symbol:
            return f"s{r}({e1},{e2})"
        else:
            return f"r{r}({e1},{e2})"

    def rules_to_prolog(self, rules):
        res = ""

        for rule in rules:
            res += self.rule_to_prolog(rule) + ".\n"
        return res

    def rule_to_prolog(self, rule, symbol=True):
        head = self.triple_to_prolog(rule['consequent'][0], symbol=symbol)

        prolog_antecedents = [self.triple_to_prolog(t, symbol=True) for t in rule['antecedents']]
        body = ','.join(prolog_antecedents)

        return f"{head} :- {body}"

    def triple_to_original(self, triple):
        e1, r, e2 = triple
        e1 = self.id_to_variable[e1] if e1 < 0 else self.id_to_entity[e1]
        e2 = self.id_to_variable[e2] if e2 < 0 else self.id_to_entity[e2]
        r = self.id_to_relation[r]

        return e1, r, e2

    def get_facts_at_cut(self, lambda_cut):
        new_facts = defaultdict(float)
        self.fact_unifications = {}
        result = []
        for fact in self.facts:
            pred_idx = fact[1]

            relevant_sims = self.symbol_predicate_similarity[:, pred_idx]
            K_cut = np.sort(relevant_sims)[::-1][:K][-1]

            cut = max(K_cut, lambda_cut)
            cut = lambda_cut

            # needs to be `>=' and not `>' because otherwise it won't work if the K most similar predicates have the same score
            for new_pred in np.where(relevant_sims >= cut)[0]:
                new_fact = copy.deepcopy(fact)
                new_fact[1] = new_pred
                new_fact = self.triple_to_prolog(new_fact, symbol=True) + '.'
                score = relevant_sims[new_pred]

                if score > new_facts[new_fact]:
                    new_facts[new_fact] = score
                    self.fact_unifications[new_fact] = f"s{self.id_to_symbol[new_pred]}<>p{self.id_to_relation[pred_idx]}"
        for fact, score in new_facts.items():
            result.append(fact + ' = ' + str(score))
        return result

    def get_rules_at_cut(self, lambda_cut):
        new_rules = defaultdict(float)
        self.rule_unifications = {}
        result = []
        for rule in self.rules:
            new_rules[self.rule_to_prolog(rule) + '.'] = 1.0
            symbol_idx = rule['consequent'][0][1]

            # We do not need rules with symbol_predicate_sim because 
            # we facts for the similar symbols are generated
            # 
            # relevant_sims = self.symbol_predicate_similarity[symbol_idx]
            # K_cut = np.sort(relevant_sims)[::-1][:K][-1]
            # cut = max(K_cut, lambda_cut)
            # for new_consequent in np.where(relevant_sims >= cut)[0]:
                # new_rule = copy.deepcopy(rule)
                # new_rule['consequent'][0][1] = new_consequent
                # new_rule = self.rule_to_prolog(new_rule, symbol=False) + '.'
                # score = relevant_sims[new_consequent]

                # if score > new_rules[new_rule]:
                    # new_rules[new_rule] = score
                    # self.rule_unifications[new_rule] = f"p{self.id_to_relation[new_consequent]},s{self.id_to_symbol[symbol_idx]}"

            relevant_sims = self.symbol_similarity[symbol_idx]

            K_cut = np.sort(relevant_sims)[::-1][:K][-1]
            cut = max(K_cut, lambda_cut)
            cut = lambda_cut


            for new_consequent in np.where(relevant_sims >= cut)[0]:
                new_rule = copy.deepcopy(rule)
                new_rule['consequent'][0][1] = new_consequent
                new_rule = self.rule_to_prolog(new_rule, symbol=True) + '.'
                score = relevant_sims[new_consequent]

                if score > new_rules[new_rule]:
                    new_rules[new_rule] = score
                    self.rule_unifications[new_rule] = f"s{self.id_to_symbol[new_consequent]}<>s{self.id_to_symbol[symbol_idx]}"

        for rule, score in new_rules.items():
            result.append(rule + ' = ' + str(score))

        return result

    def get_similarities(self):
        result = []

        sims = np.triu(self.entity_similarity)
        for i in range(sims.shape[0]):
            K_cut = np.sort(sims[i])[::-1][:K][-1]
            for j in range(sims.shape[1]):
                if i ==j or sims[i,j] <= 0:
                    continue
                else:
                    result.append(f"e{i} ~ e{j} = {sims[i,j]}")

        sims = self.symbol_predicate_similarity
        for i in range(sims.shape[0]):
            K_cut = np.sort(sims[i])[::-1][:K][-1]
            for j in range(sims.shape[1]):
                if sims[i,j] <= 0:
                    continue
                else:
                    result.append(f"s{i} ~ r{j} = {sims[i,j]}")

        sims = np.triu(self.symbol_similarity)
        for i in range(sims.shape[0]):
            K_cut = np.sort(sims[i])[::-1][:K][-1]
            for j in range(sims.shape[1]):
                if i ==j or sims[i,j] <= 0:
                    continue
                else:
                    result.append(f"s{i} ~ s{j} = {sims[i,j]}")

        #sims = np.triu(self.predicate_similarity)
        #for i in range(sims.shape[0]):
            #for j in range(sims.shape[1]):
                #if i ==j or sims[i,j] == 0:
                    #continue
                #else:
                    #result.append(f"r{i} ~ r{j} = {sims[i,j]}")


        return result

    def get_unifications(self, rules, entity_unifications):
        unifications = []

        if len(entity_unifications) > 0:
            for ent_unification in entity_unifications.split('|'):
                e1, e2 = ent_unification.split(',')
                e1_idx = int(e1.strip('e'))
                e2_idx = int(e2.strip('e'))
                unifications.append(f"e{self.id_to_entity[e1_idx]}<>e{self.id_to_entity[e2_idx]}")


        if len(rules) > 0:
            for rule in rules.split('|')[::2][1:]:
                if rule in self.rule_unifications:
                    unifications.append(self.rule_unifications[rule])
                elif rule in self.fact_unifications:
                    unifications.append(self.fact_unifications[rule])
        
        return unifications



