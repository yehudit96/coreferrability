"""
In this script we add the predicted cluster from the coreference system (event_entity_coref_ecb_plus) as a feature to the data
"""
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from ntpath import basename

import sys

sys.path.append('../../event_entity_coref_ecb_plus')
import result_analysis.clusters_analysis as ca

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

events_mention = 'corpus_Event_gold_mentions.json'
entities_mention = 'corpus_Entity_gold_mentions.json'

events_clusters = 'event_clusters_check.txt'
entities_clusters = 'entity_clusters_check.txt'

good_dir = 'good'
bad_dir = 'bad'


def main():
    """
    compute the coreference feature akk over the rules dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentions_dir', type=str, required=True, help='Directory with event and entity mentions')
    parser.add_argument('--clusters_dir', type=str, required=True, help='Clusters directory')
    parser.add_argument('--data_dir', type=str, required=True, help='tweets (good and bad) data directory')
    parser.add_argument('--save', type=bool, default=False, required=False, help='save changes')
    
    
    global args
    global error
    global no_error
    error = 0
    no_error = 0
    args = parser.parse_args()
#    for arg, value in args.get_kwargs():
#        logger.info('{}:\t{}'.format(arg, value))
    
    logger.info('adding event cluster features to data..')
    event_rule_clusters = get_rule_clusters_dict('event', args.clusters_dir, args.mentions_dir)
    add_feature(event_rule_clusters, 'event')
    logger.info('event clusters added to data')

    logger.info('adding entity clusters to data..')
    entity_rule_clusters = get_rule_clusters_dict('entity', args.clusters_dir, args.mentions_dir)
    add_feature(entity_rule_clusters, 'entity')
    logger.info('entity clusters added to data')


def get_rule_clusters_dict(mention_type, clusters_dir, mentions_dir):
    """
    Create a dictionary 
    :param mention_type: the mentuions type - event\entity
    :param clusters_dir: the directory of output clusters from coref system
    :param mention_dir: the directory pd mentions json files
    return dictionary of clusters maps to rule and tweets id
    """
    if mention_type == 'event':
        clusters_path = os.path.join(clusters_dir, events_clusters)
        mentions_path = os.path.join(mentions_dir, events_mention)
    else:
        clusters_path = os.path.join(clusters_dir, entities_clusters)
        mentions_path = os.path.join(mentions_dir, entities_mention)

    # extract clusters data to dictionary of topics
    topics = ca.extract_clusters(clusters_path)
    # extract dictionary that maps coref_chain to doc_id 
    coref_chain_to_doc_id = ca.extract_coref_to_doc_id(mentions_path)

    rule_clusters = defaultdict(dict)
    
    # create a dictionary of cluster for each id in rule
    for topic in tqdm(topics):
        rule = topic[0][0][1].split('_', 1)[0].replace('+', '_')
        try:
            ids = get_ids(topic, coref_chain_to_doc_id)
            if ids == []:
                continue
            rule_clusters[rule][ids] = topic
        except:
            print('error:{}, no error:{}'.format(error, no_error))
            raise
    return dict(rule_clusters)
    
    
def add_feature(rule_clusters, mention_type):
    """
    add the mention_type (event\entity) feature to the rule's data
    :param rule_clusters dictionaty of clusters by rule and ids
    :param mention_type clusters of entity of event
    """
    feature = 'event_clusters' if mention_type=='event' else 'entity_clusters'
    
    good_path = os.path.join(args.data_dir, good_dir)
    good_files = [os.path.join(good_path, f) for f in os.listdir(good_path)]
    
    bad_path = os.path.join(args.data_dir, bad_dir)
    bad_files = [os.path.join(bad_path, f) for f in os.listdir(bad_path)]
    
    files = bad_files + good_files
    logger.info('feature name: {}'.format(feature))
    g = 0
    missing_rules = []
    
    # iterate over all the rules in the dataset
    for rule_f in tqdm(files):
        with open(rule_f, 'rb') as f:
            data = pickle.load(f)
        rule = basename(rule_f).split('.')[0]
        
        # if the rule haven't appeared in the clusters add an empty feature
        if rule not in rule_clusters:
            missing_rules.append(rule)
            for pair in data:
                if 'features' not in pair:
                    pair['features'] = {}
                if feature not in pair['features']:
                    pair['features'][feature] = []
        else:
            # iterate over the tweets pairs and assign the matching coref cluster
            for pair in data:
                ids = pair['tweets'][0]['id'], pair['tweets'][1]['id']
                ids = tuple(sorted(ids))
                if ids in rule_clusters[rule]:
                    clusters = rule_clusters[rule][ids]
                elif ids[0] in rule_clusters[rule]:
                    clusters = rule_clusters[rule][ids[0]]
                elif ids[1] in rule_clusters[rule]:
                    clusters = rule_clusters[rule][ids[1]]
                else:
                    clusters = []
                    #logger.error('no match {} cd  for {} ({})'.format(mention_type, ids, rule))
                    
                if 'features' not in pair:
                    pair['features'] = {}
                if clusters == []:
                    logger.error('no match {} cd  for {} ({})'.format(mention_type, ids, rule))    

                if feature not in pair['features'] or (feature in pair['features'] and pair['features'][feature] == []):
                    pair['features'][feature] = clusters
                g += 1
        if args.save:
            with open(rule_f, 'wb') as f:
                pickle.dump(data, f)
                #logger.info('saved {}'.format(f))
    #print('missing rules: {}'.format(missing_rules))
    print(g)


def get_ids(clusters, coref_chain_to_doc_id):
    """
    get ids pair for clusters of topic by their coref_chain
    :param clusters: clusters to get ids for
    :param coref_chain_to_doc_id (dict) : dictionary that maps coref_chain to doc_id
    return the tweets ids
    """
    t_ids = set()
    for cluster in clusters:
        for mention in cluster:
            t_ids.update(coref_chain_to_doc_id[mention])
    if len(t_ids) == 1:
        return t_ids.pop()
    elif len(t_ids) == 0:
        print('error at clusters {}, {}'.format(clusters, t_ids))
        return []
        
    return tuple(sorted(t_ids))


if __name__ == '__main__':
    main()
