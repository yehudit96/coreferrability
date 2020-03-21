import os
import pickle
from tqdm import tqdm
import sys
sys.path.append('../../graph')
import rule_graph
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

rules_f = '../chirps_resource/27-10-19/rules.tsv'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The rules directory")
    parser.add_argument("--label", default=None, type=int, required=True, help="The rules label")
    parser.add_argument("--features_file", default=None, type=str, required=True, help="The out features file")
    
    args = parser.parse_args()
    
    logger.info('data directory:\t{}'.format(args.data_dir))
    logger.info('rule label:\t{}'.format(args.label))
    logger.info('rule features out file path:\t{}'.format(args.features_file))

    rule_features = {f.split('.')[0]:{'label':args.label,
                                      'features':{'components':[],
                                                  'chirps':[]}} 
                     for f in os.listdir(args.data_dir)}

    rules = rule_graph.extract_rules(args.data_dir)

    logger.info('extracting chirps features..')
    chirps_features(rules, rule_features)
    logger.info('extracted all chirps features')
    
    logger.info('extracting components features..')
    component_features(rules, rule_features)
    logger.info('extracted all components features')
    
    with open(args.features_file, 'wb') as f:
        pickle.dump(rule_features, f)
    logger.info('saved rule features file to {}'.format(args.features_file))

def chirps_features(rules, feats):
    with open(rules_f) as chirps_rules:
        for r in tqdm(chirps_rules):
            r1, r2, count, days = r.strip().split('\t')
            cr1, cr2 = rule_graph.cleaned_rules[r1], rule_graph.cleaned_rules[r2]
            rule = tuple(sorted([cr1, cr2]))
            if rule in rules:
                feats['_'.join(rule)]['features']['chirps'].append((int(count), int(days), r1, r2))
                

def component_features(rules, feats):
    rule_edges = rule_graph.edges_per_rules(rules)
    logger.info('start calculating componnents size')
    for rule, edges in rule_edges.items():
        rule_G = rule_graph.create_graph(edges)
        comps = rule_graph.componnents_size(rule_G)
        feats['_'.join(rule)]['features']['components'] = comps


if __name__ == '__main__':
    main()