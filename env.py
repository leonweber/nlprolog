import os
import json
import numpy as np
import logging
import copy
from pathlib import Path

import utils

import prolog
from preprocessing import load_triples
from preprocessing import load_rules


logging.basicConfig(level=logging.ERROR)
DEBUG = False


class BaseEnv:
    def __init__(self, data_dir, shuffle=True, evaluate=False, rules='random'):
        logging.debug("Setting up enviroment...")
        self.evaluate = evaluate
        split_type = 'dev' if evaluate else 'train'

        with open(os.path.join(f'{utils.BASE_PATH}/data', data_dir, f'{split_type}.json')) as f:
            self.data: list = json.load(f)
            no_support_data = []
            for d in self.data:
                new_supports = []
                for support in d['supports']:
                    new_supports.append([s.lower() for s in support])
                d['supports'] = new_supports
                d['query'] = [q.strip('?') if q == 'X' else q.lower().strip('?') for q in d['query']]
                d['candidates'] = [c.lower() for c in d['candidates']]
                d['answer'] = d['answer'].lower()
                if len(d['supports']) < 1:
                    no_support_data.append(d)

            for d in no_support_data:
                self.data.remove(d)
            if len(no_support_data):
                print(f"WARNING: Removed {len(no_support_data)} problems because of they did not have any supports. This is problematic for the evaluation!")

        self.rules = load_rules(os.path.join(f'{utils.BASE_PATH}/rules', rules))
        self.predicate_vocab = None
        self._build_vocabs()

        self.current_problem = None
        self.current_program = None
        self.n_problems_in_this_episode = 0
        self.last_obs = None
        self.shuffle = shuffle
        self.seed = 123
        self.min_depth = 0
        self.min_bs_size = 0
        self.lambda_cut = 0.5
        self.entity_tnorm = 'prod'
        self.predicate_tnorm = 'prod'
        self.max_depth = 3
        self.compute_candidates = True
        logging.debug("Done setting up environment")


    def __len__(self):
        return len(self.data)


    def _build_vocabs(self):
        self.entity_vocab = set()
        self.predicate_vocab = set()
        self.symbol_vocab = set()
        self.trainable_symbols = set()

        for problem in self.data:
            for triple in problem['supports']:
                self.entity_vocab.add(triple[0])
                self.entity_vocab.add(triple[2])
                self.predicate_vocab.add(triple[1])

        for rule in self.rules:
            for triple in rule['antecedents'] + rule['consequent']:
                if triple[1] not in self.symbol_vocab:
                    self.symbol_vocab.add(triple[1])

        for datapoint in self.data:
            obj_slot = datapoint['query'].index('X')
            sub_slot = 0 if obj_slot == 2 else 2
            goal_symbol = datapoint['query'][1]
            goal_entity = datapoint['query'][sub_slot]
            self.symbol_vocab.add(goal_symbol)
            self.entity_vocab.add(goal_entity)

            for candidate in datapoint['candidates']:
                self.entity_vocab.add(candidate)

        self.entity_vocab = sorted(self.entity_vocab)
        self.predicate_vocab = sorted(self.predicate_vocab)
        self.symbol_vocab = sorted(self.symbol_vocab)

    def set_problem(self, i):
        self.last_obs = self._get_problem(i)
        return self.last_obs


    def _get_problem(self, i):
        problem = self.data[i]
        self.current_problem = problem
        candidate_triples = []

        for candidate in problem['candidates']:
            triple = problem['query'][:]
            obj_slot = triple.index('X')
            triple[obj_slot] = candidate
            candidate_triples.append(triple)

        self.current_program = prolog.Program(self.current_problem['supports'], self.rules, candidate_triples)
        entities = [None] * len(self.current_program.entity_to_id)
        predicates = [None] * len(self.current_program.relation_to_id)
        symbols = [None] * len(self.current_program.symbol_to_id)

        for entity, i in self.current_program.entity_to_id.items():
            entities[i] = entity

        for predicate, i in self.current_program.relation_to_id.items():
            predicates[i] = predicate

        for symbol, i in self.current_program.symbol_to_id.items():
            symbols[i] = symbol

        return {'predicates': predicates, 'symbols': symbols, 'entities': entities}

    def reset(self):
        if self.shuffle:
            random = np.random.RandomState(self.seed)
            idx = random.randint(len(self.data),)
        else:
            idx = 0

        self.n_problems_in_this_episode = 1
        self.last_obs = self._get_problem(idx)
        return self.last_obs

    def step(self, action):
        self.last_action = action
        self.current_program.set_similarities(
            symbol_predicate=action['symbol_predicate_similarity'],
            entity=action['entity_similarity'],
            symbol=action['symbol_similarity']
        )

        depths, scores, rules, unifications = self._execute()

        self.n_problems_in_this_episode += 1

        return {"depths": depths, "scores": scores, "rules": rules, "unifications": unifications}



    def _run_query(self, candidate_idx=None):
        cut = prolog.LAMBDA_CUT_PRE
    

        prolog_rules = self.current_program.get_rules_at_cut(cut)
        prolog_facts = self.current_program.get_facts_at_cut(cut)
        prolog_prog = prolog_rules + prolog_facts
        prolog_prog = '\n'.join(prolog_prog)
        sims = '\n'.join(self.current_program.get_similarities())

        if DEBUG:
            debug_sims = copy.deepcopy(sims)
            debug_prog = copy.deepcopy(prolog_prog)
            for relation, id_ in self.current_program.relation_to_id.items():
                debug_sims = debug_sims.replace(f"r{id_} ", f"{relation} ")
                debug_prog = debug_prog.replace(f"r{id_}(", f"{relation}(")
            for entity, id_ in self.current_program.entity_to_id.items():
                debug_sims = debug_sims.replace(f"e{id_} ", f"{entity} ")
                debug_prog = debug_prog.replace(f"e{id_},", f"{entity},")
                debug_prog = debug_prog.replace(f"e{id_})", f"{entity})")
            for symbol, id_ in self.current_program.symbol_to_id.items():
                debug_sims = debug_sims.replace(f"s{id_} ", f"{symbol} ")
                debug_prog = debug_prog.replace(f"s{id_}(", f"{symbol}(")

            with open('similarities.txt', 'w') as f:
                f.write(debug_sims)
            with open('program.txt', 'w') as f:
                f.write(debug_prog)

        if candidate_idx is None:
            candidates = self.current_program.candidates
        else:      
            candidates = [self.current_program.candidates[candidate_idx]]
        query_str = '|'.join(self.current_program.triple_to_prolog(c, symbol=True, invert=False) + '.' for c in candidates)

        if DEBUG:
            with open('problem.json', 'w') as f:
                json.dump(self.current_problem, f)
            with open('query.txt', 'w') as f:
                f.write(query_str)

        result = prolog.query(prolog_prog, sims, query_str, entity_tnorm=self.entity_tnorm, predicate_tnorm=self.predicate_tnorm, min_depth=self.min_depth, min_bs_size=self.min_bs_size, lambda_cut=self.lambda_cut, max_depth=self.max_depth)

        # for relation, id_ in self.current_program.relation_to_id.items():
            # for r in result:
                # r[2] = r[2].replace(f"r{id_} ", f"{relation} ")
        # for entity, id_ in self.current_program.entity_to_id.items():
            # for r in result:
                # r[2] = r[2].replace(f"e{id_},", f"{entity},")
                # r[2] = r[2].replace(f"e{id_})", f"{entity})")
        # for symbol, id_ in self.current_program.symbol_to_id.items():
            # for r in result:
                # r[2] = r[2].replace(f"s{id_}(", f"{symbol}(")


        return result

    def _execute(self):
        raise NotImplementedError


class QAEnv(BaseEnv):

    def _execute(self):

        candidates = self.current_problem['candidates']

        for i, candidate in enumerate(candidates):
            if candidate == self.current_problem['answer']:
                correct_idx = i

        if correct_idx is None:
            return -1., 0, False, np.array([]), ""

        idx = None
        if not self.compute_candidates:
            idx = correct_idx
            correct_idx = 0

        result = self._run_query(idx)

        depths = []
        scores = []
        rules = []
        unifications = []
        for i, (score, depth, rule, unification) in enumerate(result):

            depths.append(depth)
            scores.append(score)
            rules.append(rule)
            unifications.append(unification)

        scores = np.array(scores)

        pred_idx = np.argmax(scores)
        correct = np.argmax(scores) == correct_idx

        return depths, scores, rules, unifications


