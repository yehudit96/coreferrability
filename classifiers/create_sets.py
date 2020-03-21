from rule_feature_vector import extract_rules_vectors

from random import shuffle
import pickle
import json 

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

bad_dir = 'bad_events_all'
good_dir = 'good_events_all'

rules_features = '../data/tweets/{}/rules_features.pk'

data_dir = '../data/tweets/{}/{}'

features_file = 'features.json'

def main():
    """
    create pickled file for test\train\dev split of features vectors
    """
    train_feat = extract_set_vectors('train')
    dev_feat = extract_set_vectors('dev')
    
    assert train_feat == dev_feat
    
    logger.info('saving features list')
    with open(features_file, 'w') as ff:
        json.dump(dev_feat, ff)
    logger.info('saved features list to {}'.format(features_file))

    
def extract_set_vectors(set):
    logger.info('extracting rules vectors for {}'.format(set))
    good_rules_vectors, good_features = extract_rules_vectors(data_dir.format(set, good_dir), rules_features.format(set))
    bad_rules_vectors, bad_features = extract_rules_vectors(data_dir.format(set, bad_dir), rules_features.format(set))
    
    assert good_features == bad_features
   
    good_rules_vectors = [rv + tuple([1]) for rv in good_rules_vectors]
    bad_rules_vectors = [rv + tuple([0]) for rv in bad_rules_vectors]
    
    rules_vectors = good_rules_vectors + bad_rules_vectors
    
    shuffle(rules_vectors)

        
    logger.info('saving {} vectors'.format(set))
    with open(set, 'wb') as f:
        pickle.dump(rules_vectors, f)
    logger.info('saved vectors to {}'.format(set))
    return bad_features
    
if __name__ == '__main__':
    main()