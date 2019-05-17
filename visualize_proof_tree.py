import re

from graphviz import Digraph
import json


class Node:
    def __init__(self, id, parent, rule):
        self.id = id
        self.parent = parent
        self.rule = rule
        self.children = []

    def get_leftmost_active_child(self):
        for child in self.children:
            if not child.rule.is_done():
                return child
        return None

    def __str__(self):
        return str(self.rule)

    def __repr__(self):
        self.__str__()



class Rule:
    def __init__(self, head, body):
        self.head = head
        self.body = body
        self.status = [False for atom in body]

    def __str__(self):
        return self.head + " :- " + ",".join(self.body)

    @staticmethod
    def parse(rule):
        split = rule.split(':-')
        head = split[0].strip()
        body = split[1] if len(split) > 1 else ""

        body_atoms = re.findall(r'(.+?\(.+?\)\s*)[,.]', body)
        body_atoms = [b.strip() for b in body_atoms]

        return Rule(head, body_atoms)

    @staticmethod
    def signature(atom):
        return atom.split('(')[0] + '/' + str(atom.count(',') + 1)

    def is_done(self):
        return False not in self.status


def build_tree(rules):
    split = rules.split('|')
    rules = split[::2]
    scores = split[1::2]

    root = Node(0, None, Rule.parse(rules[0]))
    edges = []
    nodes = [root]

    for i, rule in enumerate(rules[1:], start=1):
        rule = Rule.parse(rule)
        current_node = root
        next_node = current_node.get_leftmost_active_child()

        while next_node is not None:
            current_node = next_node
            next_node = next_node.get_leftmost_active_child()

        rule_node = Node(i, current_node, rule)
        current_node.children.append(rule_node)

        nodes.append(rule_node)
        edges.append((current_node.id, i, scores[i-1]))

        current_node = rule_node
        while current_node != root and current_node.rule.is_done():
            for i, parent_body_elem in enumerate(current_node.parent.rule.body):
                if not current_node.parent.rule.status[i]:
                    sig1 = Rule.signature(parent_body_elem)
                    sig2 = Rule.signature(current_node.rule.head)
                    assert sig1 == sig2, f"sig1: {sig1}, sig2: {sig2}"
                    current_node.parent.rule.status[i] = True
                    break
            current_node = current_node.parent


    return nodes, edges

def plot(nodes, edges, name="plot"):
    dot = Digraph(name=name)
    for node in nodes:
        dot.node(str(node.id), str(node.rule))

    for parent_id, child_id, score in edges:
        dot.edge(str(parent_id), str(child_id), label=score)

    dot.render(name, view=False)
    return dot


if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    idx = int(sys.argv[2])

    with open(fname) as f:
        rules = json.load(f)

    rules = rules[idx]
    # rules = [':'.join(r.split(':')[1:]) for r in rules]
    # rules = 'is_afraid_of(gertrude,wolves).|1.0|is_afraid_of(X,Z) :- is_afraid_of(Y,Z), is_a(X,Y).|1.0|is_afraid_of(X,Z) :- is_afraid_of(Y,Z), is_a(X,Y).|1.0|is_afraid_of(animals, wolves).|1.0|is_a(mice, animals).|0.74|is_a(gertrude, mouse).'
    nodes, edges = build_tree(rules)
    plot(nodes, edges)

