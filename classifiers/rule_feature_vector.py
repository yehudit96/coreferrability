import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import sys
sys.path.append('/home/nlp/yehudit/projects/coreferrability/rules_ranking/')
from rank_utils import convert_date
from datetime import timedelta
from tweet_features_vector import extract_pair_features
from sklearn.feature_extraction import DictVectorizer

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

bad_dir = 'bad'
good_dir = 'good'

NE_ts = 0.26
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The rules directory")
    parser.add_argument("--rules_features", default=None, type=str, required=True, help="The rules features pk file")
    parser.add_argument("--vectors_path", default=None, type=str, required=True, help="Features vectores out file path")
    args = parser.parse_args()
    
    rules_vectors, features = extract_rules_vectors(args.data_dir, args.rules_features)
    
    with open(args.vectors_path, 'wb') as f:
        pickle.dump(rules_vectors, f)
    
    with open('features', 'w') as f:
        json.dump(features, f)

    
def extract_rules_vectors(data_dir, rules_features):
    logger.info('rule feature extraction starting..')
    bad_events_path = os.path.join(data_dir, bad_dir)
    good_events_path = os.path.join(data_dir, good_dir)
    files = [(os.path.join(bad_events_path, f), 0) for f in os.listdir(bad_events_path)]
    files += [(os.path.join(good_events_path, f), 1) for f in os.listdir(good_events_path)]


    rules = []
    
    with open(rules_features, 'rb') as f:
        rules_features = pickle.load(f)
    
    for rule_f, label in tqdm(files):
        rule = os.path.basename(rule_f).split('.')[0]
        rule_features = extract_features(rule_f, rules_features)
        rules.append((rule, rule_features, label))
        
    print(rule_features)
    
    rules_names, rules_features, rules_labels = zip(*rules)
    vec = DictVectorizer()
    v = vec.fit_transform(rules_features).toarray()
    
    rules_v = list(zip(rules_names, v, rules_labels))
    
    print(vec.get_feature_names())
    print(len(vec.get_feature_names()), len(v[0]))
    print(len(rules_v))
    return rules_v, vec.get_feature_names()
        
    
def extract_features(rule_path, rules_features):
    with open(rule_path, 'rb') as f:
        rule_data = pickle.load(f)

    rule_features = {}
    pairs_features = []
    for pair in rule_data:
        pairs_features.append(extract_pair_features(pair))

    rule_features['pairs_num'] = len(pairs_features)
    rule_features['day_num'] = len(set().union(*[feat['days'] for feat in pairs_features]))
    
    #  rule_features['NE_avg'] = sum(feat['NE'] for feat in pairs_features)/len(pairs_features)
    rule_features['NE_{}'.format(NE_ts)] = sum(1 for feat in pairs_features if feat['NE'] >= NE_ts)  #  /len(pairs_features)
    rule_features['NE_{}_avg'.format(NE_ts)] = sum(feat['NE'] for feat in pairs_features if feat['NE'] >= NE_ts)/len(pairs_features) if len(pairs_features) else 0
    
    #  rule_features['NE_sum'] = sum(feat['NE'] for feat in pairs_features)
    
    rule_features['event_pc'] = sum(1 for feat in pairs_features if feat['event_clusters_type'] == 'pc')  # /len(pairs_features)
    rule_features['event_ipc'] = sum(1 for feat in pairs_features if feat['event_clusters_type'] == 'ipc')
    rule_features['entity_pc'] = sum(1 for feat in pairs_features if feat['entity_clusters_type'] == 'pc')  #  /len(pairs_features)
    rule_features['entity_ipc'] = sum(1 for feat in pairs_features if feat['entity_clusters_type'] == 'ipc')  #  /len(pairs_features)
    rule_features['entity_wc'] = sum(1 for feat in pairs_features if feat['entity_clusters_type'] == 'wc')  #  /len(pairs_features)
    
    has_NE = any(map(lambda f:f['has_NE'], pairs_features))
    rule_features['NE_{}_event_pc'.format(NE_ts)] = sum(1 for feat in pairs_features if feat['event_clusters_type'] == 'pc' and feat['NE'] >= NE_ts) if has_NE else -1#  /len(pairs_features)
    #  rule_features['time_delta_max'] = max(feat['time_delta'] for feat in pairs_features)
    #  rule_features['time_delta_avg'] = sum(feat['time_delta'] for feat in pairs_features)/len(pairs_features)
    
    rule_features['in_clique'] = sum(1 for feat in pairs_features if feat['in_clique'])  #  /len(pairs_features)
    #  rule_features['in_clique_size_max'] = max(feat['in_clique_size'] for feat in pairs_features)
    

    features = rules_features[os.path.basename(rule_path).split('.')[0]]['features']
    rule_features['chirps_days'] = sum(c[1] for c in features['chirps'])
    rule_features['chirps_num'] = sum(c[0] for c in features['chirps'])
    rule_features['chirps_rules_num'] = len(features['chirps'])
    rule_features['chirps_max_rank'] = max(c[0]*(1 + c[1] / 500.0) for c in features['chirps'])
    
    #  rule_features['chirps_my_score'] = len(pairs_features)*(1 + rule_features['day_num'] / 500.0)
    if features['components']:
        #  rule_features['component_max'] = max(features['components'])
        #  rule_features['component_min'] = min(features['components'])
        rule_features['component_avg'] = sum(features['components'])/len(features['components'])
    else:
        #  rule_features['component_max'] = -1
        #  rule_features['component_min'] = -1
        rule_features['component_avg'] = 0
    
    rule_features['component_num'] = len(features['components'])
    


    return rule_features


if __name__ == '__main__':
    main()