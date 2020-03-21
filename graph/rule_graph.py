"""
Create a graph based align tweets pairs (ids) per rule.
"""
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import os
import sys
sys.path.append('/home/nlp/yehudit/projects/coreferrability/rules_ranking/')
from rank_utils import AP
import pickle
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


instances_f = '/home/nlp/yehudit/projects/coreferrability/data/chirps_resource/27-10-19/instances.tsv'
good_rules_path = '../data/tweets/good_tweets'
bad_rules_path = '../data/tweets/bad_tweets'
save_to = '../rules_ranking/connected_components.txt'
cleaned_rules_file = '/home/nlp/yehudit/projects/coreferrability/data/chirps_resource/27-10-19/cleaned_rules.pk'
with open(cleaned_rules_file, 'rb') as f:
    cleaned_rules = pickle.load(f)

def main():
    """
    Create graphs for all good and bad rules.
    Calculate a avarage component size.
    Save ranked rules list to save_to file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The data pickle files directory")
    parser.add_argument("--data_dir2", default=None, type=str, required=False, help="The 2nd data pickle files directory (optional)")
    
    args = parser.parse_args()
    rules_paths = [args.data_dir]
    if args.data_dir2:
        rules_paths.append(args.data_dir2)
    rules = []
    for path in rules_paths:
        rules += extract_rules(good_rules_path)
    rule_edges = edges_per_rules(rules)
    rule_avg_comp_size = []
    logger.info('start calculating componnents size')
    for rule, edges in rule_edges.items():
        rule_G = create_graph(edges)
        avg_size = componnents_size(rule_G)
        rule_type = 1 if rule in good_rules else 0
        rule_avg_comp_size.append(['\t'.join(rule), avg_size, rule_type])
    rule_avg_comp_size = sorted(rule_avg_comp_size, key=lambda x:x[1], reverse=True)
    rule_rank = [r[2] for r in rule_avg_comp_size]
    graph_AP = AP(rule_rank)
    print(graph_AP)
    save_ranked_rules(rule_avg_comp_size)


def extract_rules(path):
    """
    extract all the rules in a path
    Parameters:
    path (str): path with rules files
    
    Returns:
    files (list): list of rules in the path
    """
    rules = [tuple(sorted(f.replace('.pk', '').split('_'))) for f in os.listdir(path)]
    return rules


def edges_per_rules(rules):
    """
    create a dictionary of edges (value) per rule (key), the edges exported from the instances file.
    Each instance is added to its rule's set in the rule_edges dictionary.
    
    Parameters:
    rules (list): list of rules
    
    Returns:
    rules_edges (dict): dictionary 
    """
    logger.info('start reading instances file')
#    with open(cleaned_rules_file, 'rb') as f:
#        cleaned_rules = pickle.load(f)
    rules = [tuple(r) for r in rules]
    rule_edges = defaultdict(set)
    with open(instances_f) as instf:
        for l in tqdm(instf):
            data = l.split('\t')
            r1, r2 = data[2], data[7]
            ids = (data[0], data[5])
            c1, c2 = cleaned_rules[r1], cleaned_rules[r2]
            cr = tuple(sorted([c1, c2]))
            if cr in rules:
                rule_edges[cr].add(ids)
            #for rule in rules:
            #    if is_in_rule(r1, r2, rule):
            #        rule_edges[rule].add(ids)
            #        break
    logger.info('end reading instances file')
    return rule_edges


def is_in_rule(r1, r2, rule):
    """
    check if giver predicate pair - r1, r2 in rule.
    
    Parameters:
    r1 (str): first predicate of the instance
    r2 (str): second predicate of the instance
    rule (tuple(str)): rule to check on.
    
    Return:
    (bool): wheather r1, r2 in rule or not
    """
    if (rule[0] in r1 and rule[1] in r2) or (rule[1] in r1 and rule[0] in r2):
        return True
    return False


def create_graph(edges):
    """
    create networkx graph from edges (nodes pairs)
    
    Parameters:
    edges (list(tuple(str))): list of edges
      
    Returns:
    G (graph): graph made out of edges list
    """
    G = nx.Graph()
    for e1, e2 in edges:
        G.add_edge(e1, e2)
    return G


def componnents_avg_size(G):
    """
    calculate the avg components size
    
    Parameters:
    G (graph): the graph to be calculated
    
    Returns:
    (float): avarage size of a component in the graph
    """
    return sum([len(c) for c in nx.connected_components(G) if len(c)>2])/sum([1 for c in nx.connected_components(G) if len(c)>2])


def componnents_size(G):
    """
    make a list of components size    
    Parameters:
    G (graph): the graph to be calculated
    
    Returns:
    (float): sizes of the component in the graph
    
    """
    return [len(c) for c in nx.connected_components(G) if len(c)>2]


def save_ranked_rules(ranked_rules):
    """
    save the re-ranked rules (following abg size components) to save_to file
    
    Parameters:
    ranked_rules (list(tuple(str, float))): sorted raked rules list (descending) by avg component size
    
    Return:
    
    """
    str_rules = ['\t'.join(map(lambda x: str(x), r)) for r in ranked_rules]
    with open(save_to, 'w') as f:
        f.write('\n'.join(str_rules))
    

if __name__ == '__main__':
    main()
