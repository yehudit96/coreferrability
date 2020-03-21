import os
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import sys
sys.path.append('../rules_ranking/')
from rank_utils import convert_date

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    create feature vector per tweets, 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The tweets directory")
    args = parser.parse_args()
    logger.info('start extracting features for tweets')
    logger.info('data directory:\t{}'.format(args.data_dir))
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    logger.info('running on {} files'.format(len(files)))
    
    for data_file in files:
        vectors = rule_vactors(data_file)    
    logger.info('successfuly extracted features vector for {}/{} files'.format(len(files), len(files)))


def rule_vectors(data_file):
    features = []
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    for pair in data:
        pair_f = extract_pair_features(pair)
        del pair_f['days']
        features.append(pair_f)
    
    vec = DictVectorizer()
    v = vec.fit_transform(features).toarray()
    return v
                    
                    
def extract_pair_features(pair):
    features = pair['features']
    tweet1, tweet2 = pair['tweets']
    features_v = {}
    try:
        features_v['NE'] = features['NE'] if 'NE' in features else 0
        features_v['has_NE'] = 'NE' in features
        if not features['in_clique']:
            features_v['in_clique'] = 0
            features_v['in_clique_size'] = 0
        else:
            features_v['in_clique'] = 1
            features_v['in_clique_size'] = features['in_clique'][1]
        
        features_v['event_clusters_num'] = len(features['event_clusters'])
        if event_perfect_clusters(features['event_clusters']):
            features_v['event_clusters_type'] = 'pc'
        else:
            features_v['event_clusters_type'] = 'ipc'
        
        features_v['entity_clusters_num'] = len(features['entity_clusters'])
        if entity_perfect_clusters(features['entity_clusters']):
            features_v['entity_clusters_type'] = 'pc'
        elif entity_imperfect_clusters(features['entity_clusters']):
            features_v['entity_clusters_type'] = 'ipc'
        else:
            features_v['entity_clusters_type'] = 'wc'
        
        features_v['time_delta'] = (convert_date(tweet1['created_at']) - convert_date(tweet2['created_at'])).total_seconds()
        features_v['days'] = set([convert_date(tweet1['created_at']).date(), convert_date(tweet2['created_at']).date()])
    except:
        print('error in {}'.format(pair))
        raise
    return features_v

def event_perfect_clusters(clusters):
    return len(clusters) == 1 and len(clusters[0]) == 2
    

def entity_perfect_clusters(clusters):
    if len(clusters) > 2:
        return False
    for cluster in clusters:
        if len(cluster) > 2:
            return False
        cc = set([coref_chain for mention, coref_chain in cluster])
        if len(cc) != 1:
            return False
    return True
    
def entity_imperfect_clusters(clusters):
    for cluster in clusters:
        cc = set([coref_chain for mention, coref_chain in cluster])
        if len(cc) != 1:
            return False
    return True
    
    
if __name__ == '__main__':
    main()
