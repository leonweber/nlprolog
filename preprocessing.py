#!/usr/bin/env python3
import os
import string
import collections
import sys
import json

from tqdm import tqdm
import re
from joblib import Parallel, delayed

import itertools
import pexpect
from nltk import ToktokTokenizer
from nltk import sent_tokenize
from nltk.parse.corenlp import CoreNLPParser
import logging
import spacy
from spacy.matcher import PhraseMatcher
import networkx as nx

logger = logging.getLogger(os.path.basename(sys.argv[0]))

Atom = collections.namedtuple("Atom", ["predicate", "arguments"])
VAR_PREFIX = "!__VAR__!"
PRONOUNS = {"he", "she", "it"}

TYPE_BLACKLIST = {'DATE', 'TIME', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'MONEY'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s\s*', ' ', text)
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r'\s*,\s*$', '', text)

    return text


class TripleExtractorBase:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def text_to_triples(self, text, *args, **kwargs):
        raise NotImplementedError

    def get_triples(self, texts):
        triples = set()
        for text in texts:
            triples.update(self.text_to_triples(text))
        return triples

    def get_vocab(self, texts):
        vocab = set()
        triples = self.get_triples(texts)
        for e1, rel, e2 in triples:
            vocab.update(self.tokenizer(e1))
            vocab.update(self.tokenizer(rel))
            vocab.update(self.tokenizer(e2))
        return vocab


class MinieExtractor(TripleExtractorBase):
    def __init__(self, dicts, path='resources/minie.jar'):
        '''
            Initializes Minie
            :params
                path: Path to minie jar
        '''
        dicts = ['/minie-resources/' + dictname for dictname in dicts]
        dicts = ';'.join(dicts)
        cmd = f'java -jar {path} -m dictionary --dict {dicts}'
        self.cmd = cmd
        self.process = None

        self.restart_process()


    def restart_process(self):
        self.process = pexpect.spawn(self.cmd)
        self.process.setecho(False)
        self.process.expect('Setup finished, ready to take input sentence:', None)
        self.process.expect('\n')

    def text_to_triples(self, text, *args, **kwargs):
        # clear any pending output
        try:
            self.process.read_nonblocking(2048, 0)
        except:
            pass

        all_triples = []

        for sent in sent_tokenize(text):
            self.process.sendline(sent)
            self.process.waitnoecho()  # remove this if not working
            timeout = 5 + len(text) / 20.0

            i = self.process.expect(['\r\n\r\n', pexpect.TIMEOUT, pexpect.EOF], timeout)
            if b'Exception' in self.process.before:
                self.restart_process()
                continue
            else:
                results = self._parse_output(self.process.before)
                all_triples += [r['triple'] for r in results if 'factuality' in r and '-' not in r['factuality']]
        return all_triples

    @staticmethod
    def _parse_output(output):
        parses = []
        output = output.decode()
        facts = output.split('\r\n')[2:]
        #print(output)
        if len(facts) == 0 or 'No extraction found' in facts[0]:
            return []

        for fact in facts:
            match = re.match(r'\((.*?)\)(.*)', fact)
            if match is None:
                continue
            groups = match.groups()
            triple = tuple(groups[0].split(';'))
            if len(triple) != 3:
                continue
            parse = {'triple': triple}
            for group in groups[1:]:
                group = group.strip('[').strip(']')
                s = group.split('=')
                try:
                    parse[s[0]] = s[1]
                except IndexError:
                    continue


            parses.append(parse)
        return parses


def make_title(doc, nlp):
    text = []
    for tok in doc:
        text.append(tok.text[0].upper() + tok.text[1:])
    return nlp(" ".join(text))


def truncate(doc, nlp):
    return nlp(str(doc[:9]))


def filter_spans(spans, start, end):
    filtered_spans = [s for s in spans if s.start >= start and s.stop <= end]
    sorted_spans = sorted(filtered_spans, key=lambda x: x.start)
    merged_spans = []
    
    for later_span in sorted_spans:
        if len(merged_spans) == 0:
            merged_spans.append(later_span)
        else:
            earlier_span = merged_spans.pop()
            if later_span.stop in earlier_span:
                new_span = range(earlier_span.start, max(earlier_span.stop, later_span.stop))
                merged_spans.append(new_span)
            else:
                merged_spans.append(earlier_span)
                merged_spans.append(later_span)

    return merged_spans


class SpacyExtractor(TripleExtractorBase):


    def __init__(self):
        #self.nlp = spacy.load('en_coref_md')
        self.nlp = spacy.load('en')


    def text_to_triples(self, text, query_ent, candidate_ents):

        triples = set()
    
        doc = self.nlp(text)
        ents = [range(e.start, e.end) for e in doc.ents if e[0].ent_type_ not in TYPE_BLACKLIST]

        matcher = PhraseMatcher(self.nlp.vocab)
        coref = {}

        def add_to_ents(matcher, doc, i, matches):
            match_id, start, end = matches[i]
            ent_range = range(start, end)
            ents.append(ent_range)


        matcher.add(0, add_to_ents, truncate(self.nlp(query_ent), self.nlp))
        matcher.add(1, add_to_ents, truncate(self.nlp(query_ent.lower()), self.nlp))
        matcher.add(2, add_to_ents, truncate(self.nlp(query_ent.upper()), self.nlp))
        matcher.add(3, add_to_ents, truncate(make_title(self.nlp(query_ent), self.nlp), self.nlp))
        i = 3

        for cand_ent in candidate_ents:
            matcher.add(i+1, add_to_ents, truncate(self.nlp(cand_ent), self.nlp))
            matcher.add(i+2, add_to_ents, truncate(self.nlp(cand_ent.lower()), self.nlp))
            matcher.add(i+3, add_to_ents, truncate(self.nlp(cand_ent.upper()), self.nlp))
            matcher.add(i+4, add_to_ents, truncate(make_title(self.nlp(cand_ent), self.nlp), self.nlp))
            i += 4

        # matcher(doc)
        # if doc._.has_coref:
            # for cluster in doc._.coref_clusters:
                # for mention in cluster.mentions:
                    # ents.append(range(mention.start, mention.end))

        for sent in doc.sents:
            sent_ents = filter_spans(ents, sent.start, sent.end)
            for i, e1 in enumerate(sent_ents):
                for e2 in sent_ents[i+1:]:
                    pred = []
                    left = [t.string.strip() for t in doc[sent.start:e1.start]]
                    mid = [t.string.strip() for t in doc[e1.stop:e2.start]]
                    right = [t.string.strip() for t in doc[e2.stop:sent.end]]
                    pred = left + ["__ENT1__"] + mid + ["__ENT2__"] + right
                    e1_span = doc[e1.start:e1.stop]
                    e2_span = doc[e2.start:e2.stop]

                    # if doc._.has_coref and len(e1_span) == 1 and e1_span[0]._.in_coref:
                            # e1_span = e1_span[0]._.coref_clusters[0].main

                    # if doc._.has_coref and len(e2_span) == 1 and e2_span[0]._.in_coref:
                            # e2_span = e2_span[0]._.coref_clusters[0].main

                    triple = (e1_span.string.strip(), " ".join(pred), e2_span.string.strip())
                    triples.add(triple)

        return list(triples)



class CoreNLPChunkingExtractor(TripleExtractorBase):

    def __init__(self, tokenizer=None):
        tokenizer = tokenizer or ToktokTokenizer()
        super(CoreNLPChunkingExtractor, self).__init__(tokenizer)
        self.parser = CoreNLPParser()

    def has_vp(self, parse):

        stack = [parse]
        while len(stack) > 0:
            parse = stack.pop()

            try:
                if parse.label() == 'VP':
                    return True
                else:
                    for child in parse:
                        stack.append(child)
            except AttributeError:
                continue
        return False

    def get_entities(self, parse):
        text = []
        entity_indices = []
        stack = []
        stack.append(parse)

        while len(stack):
            parse = stack.pop()

            if type(parse) == str:
                text.append(parse.strip())
            elif parse.label() == 'NP' and not self.has_vp(parse):
                entity_indices.append(len(text))
                text.append(' '.join(parse.flatten()).strip())
            else:
                for child in reversed(parse):
                    stack.append(child)
        return text, entity_indices

    def text_to_triples(self, text, *args, **kwargs):
        text = ''.join(filter(lambda char: char in string.ascii_letters + ',. !?' + string.digits, text))

        parse = self.parser.parse_one([text])
        text, entity_indices = self.get_entities(parse)
        triples = []

        for i, entity1_idx in enumerate(entity_indices):
            for entity2_idx in entity_indices[i+1:]:
                entity1 = text[entity1_idx]
                entity2 = text[entity2_idx]
                predicate = ' '.join(text[entity1_idx+1:entity2_idx])
                
                triple = (entity1, predicate, entity2)
                triple = tuple(clean_text(t) for t in triple)

                if all(len(x) for x in triple):
                    triples.append(triple)
                    logging.debug(f"Generated triple: {triple}")

        return triples


def parse_prolog_atom(atom):
    predicate = atom.split('(')[0].strip()
    rest = atom.split('(')[1].strip()
    e1 = rest.split(',')[0].strip()
    e2 = rest.split(',')[1].strip().strip(')').strip()

    return e1, predicate, e2


def load_rules(fname):
    rules = []
    print("Rules:", fname)
    with open(fname) as f:
        for line in f:
            rule = {}
            line = line.strip()
            if not len(line):
                continue

            split = line.split(':-')
            head = split[0].strip()
            body = split[1].strip()
            rule['consequent'] = [parse_prolog_atom(head)]

            body_atoms = re.findall(r'(.+?\(.+?\)\s*)[,.]', body)
            rule['antecedents'] = [parse_prolog_atom(b) for b in body_atoms]
            rules.append(rule)

    return rules


def load_triples(fname):
    all_triples = []
    current_triples = []
    previous_subjects = []
    with open(fname) as f:
        for line in f:
            line = line.lower()
            line = line.strip()
            if line == '':
                if len(current_triples) > 0:
                    all_triples.append(current_triples)
                    current_triples = []
                    previous_subjects = []
            else:
                triple = line.split('|')
                if triple[0] in PRONOUNS:
                    for subject in previous_subjects: 
                        current_triples.append(subject, triple[1], triple[2])
                current_triples.append(triple)
    return all_triples


def get_goals(query, predicate_substitutions):
    goals = []
    predicate = query['query'].split()[0].lower()
    argument = " ".join(query['query'].split()[1:]).lower()

    predicates = predicate_substitutions.get(predicate, [predicate.replace("_", " ")])
    for predicate, candidate_position in predicates:
        for candidate in query['candidates']:
            if candidate_position ==0:
                goals.append((Atom(predicate, [candidate, argument]), candidate_position))
            else:
                goals.append((Atom(predicate, [argument, candidate]), candidate_position))
    return goals



def rule_to_atom(rule, predicate_substitutions):
   head = Atom(rule['consequent'][0][1], [rule['consequent'][0][0], rule['consequent'][0][2]])
   body = [Atom(i[1], [i[0], i[2]]) for i in rule['antecedents']]
   rule = [head] + body
   new_rule = [] 
   for atom in rule:
       new_args = []
       for argument in atom.arguments:
           if argument.isupper():
               argument = VAR_PREFIX + argument
           new_args.append(argument)
       if atom.predicate in predicate_substitutions:
           new_pred = predicate_substitutions[atom.predicate][0][0]
       else:
           new_pred = atom.predicate.replace('_', ' ')

       new_rule.append(Atom(new_pred, new_args))

   return new_rule


def clean_raw_text(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\"', '', text)
    text = re.sub(r'\:', '', text)

    return text



def construct_kbs(triples, rules, predicate_substitutions):
    kbs = []
    for kb_triples in triples:
        kb = []
        for triple in kb_triples:
            kb.append( [Atom(triple[1], [triple[0], triple[2]])] )
        for rule in rules:
            kb.append(rule_to_atom(rule, predicate_substitutions))
        kbs.append(kb)

    return kbs

def process(prob):
    extractor = SpacyExtractor()
    new_supps = []
    query_ent = ' '.join(prob['query'].split()[1:])
    if len(query_ent.strip()) == 0:
        return []
    for support in prob['supports']:
        triples = extractor.text_to_triples(clean_raw_text(support), query_ent, prob['candidates'])
        new_supps += triples
    return new_supps


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        problems = json.load(f)
    new_supps = Parallel(n_jobs=10, verbose=10)([delayed(process)(prob) for prob in problems])

    for prob, supps in zip(problems, new_supps):
        prob['supports'] = supps


    with open(sys.argv[2], 'w') as f:
        json.dump(problems, f)
