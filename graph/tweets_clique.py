import networkx as nx
import pickle
import os
import itertools
from tqdm import tqdm
from collections import defaultdict
import logging
"""
In this script-
1. create_cliques - we create a graph out of all tweets pairs from chirps - instances file.
    Node - tweet id
    Edge - two tweets that appears together in instances file.
    Then we extract all the cliques (size > 2), and save the clique as list of ids in clique_f
2. create_clique_dict - to feture use, if we want to check if two tweets are in the same clique of not, we create a
    dictionary that contains all the possible tweets pair from all the cliques, the dictionary saved in clique_dict_f.
3. is_in_clique - the function gets 2 ids a return weather the tweets are in the same clique or not, by the dictionary created in create_clique_dict.

"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


instances_f = '../data/chirps_resource/27-10-19/instances.tsv'
clique_f = '/home/nlp/yehudit/projects/coreferrability/graph/tweets_clique.pk'
clique_dict_f = '/home/nlp/yehudit/projects/coreferrability/graph/tweets_clique_dict.pk'

clique_exists = True

if not os.path.exists(clique_f):
    logger.info('the clique file {} doesn\'t exist. In order to use is_in_clique function you need to create the cliques file with create_cliques funtion'.format(clique_f))
    clique_exists = False
else:
    with open(clique_f, 'rb') as f:
        tweets_cliques = pickle.load(f)
    with open(clique_dict_f, 'rb') as f:
        tweets_cliques_dict = pickle.load(f)


def main():
    cliques = create_cliques()
    create_clique_dict(cliques)
    
def create_cliques():
    logger.info('reading from {}'.format(instances_f))
    edges = set()
    with open(instances_f) as instf:
        for l in tqdm(instf):
            data = l.split('\t')
            ids = (data[0], data[5])
            edges.add(ids)
    
    
    G = nx.Graph()
    
    logger.info('creating graph from {} edges'.format(len(edges)))
    for edge in tqdm(edges):
        G.add_edge(edge[0], edge[1])
    
    logger.info('extracting clique from the graph...')
    cliques = [c for c in list(nx.clique.find_cliques(G)) if len(c) > 2]
    logger.info('extracted {} cliques'.format(len(cliques)))
    
    logger.info('saving cliques to {}'.format(clique_f))
    with open(clique_f, 'wb') as f:
        pickle.dump(cliques, f)
    return cliques


def create_clique_dict(cliques_list):
    logger.info('creating cliques dictionary...')
    is_clique = defaultdict(bool)
    for clique in tqdm(cliques_list):
        for pair in list(itertools.combinations(clique, 2)):
            s_pair = tuple(sorted(pair))
            is_clique[s_pair] = (True, len(clique))
    
    logger.info('saving clique dictionary to {}'.format(clique_dict_f))
    with open(clique_dict_f, 'wb') as f:
        pickle.dump(is_clique, f)
    #return is_clique


def is_in_clique(id1, id2):
    if not clique_exists:
        raise Exception('the cliques file doesn\'t exist. In order to use this function you should create cliques file using create_cliques funtion')
    ids = tuple(sorted([id1, id2]))
    return tweets_cliques_dict[ids]


if __name__ == '__main__':
    main()