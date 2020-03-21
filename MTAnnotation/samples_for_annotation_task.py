import pickle
import os
import codecs
from collections import defaultdict
from random import sample, shuffle
from tqdm import tqdm
import re
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

rules_instances_path = '../data/chirps_resource/27-10-19/rules_instances/rules_chunk'

with open(rules_instances_path, 'rb') as f:
    rules_instances = pickle.load(f)

data_dirs = {'train': '../data/tweets/KianTrain/',
             'dev': '../data/tweets/KianDev/',
             'test': '../data/tweets/KianTest/',
             'eval': '../data/tweets/eval_rules/'}

dirs = ['good', 'bad']

def main():
    """
    create samples files for the annotation task
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split",
                        default=None,
                        type=str,
                        required=True,
                        help="The rules directory")
    parser.add_argument("--annotated_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The positive annotated rules directory")
    parser.add_argument('--out',
                        default=None,
                        type=str,
                        required=True,
                        help="the samples csv files dir")
    args = parser.parse_args()

    # get rules for annotation
    neg_rules = remove_dup(args.data_split.lower())
    
    logger.info('# of not annotated samples: {}'.format(len(neg_rules)))
    # get annotated rules (positives to balance the data, negative for validation)
    pos_annotated_rules = sample(os.listdir(os.path.join(args.annotated_dir, 'good_tweets')), 30)
    neg_annotated_rules = sample(os.listdir(os.path.join(args.annotated_dir, 'bad_tweets')), 20)
    
    rules = set(neg_rules + pos_annotated_rules + neg_annotated_rules)
    
    logger.info('# of annotated samples: {}'.format(len(rules) - len(neg_rules)))
    
    is_annotated = {extract_rule(rule): None for rule in neg_rules}
    is_annotated.update({extract_rule(rule): False for rule in neg_annotated_rules})
    is_annotated.update({extract_rule(rule): True for rule in pos_annotated_rules})
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    logger.info('start creating samples')
    
    title = ['p1', 'p2', 'org_rule_1_1', 'org_rule_1_2', 'org_rule_2_1', 'org_rule_2_2', 'org_rule_3_1', 'org_rule_3_2',
             'inst1_1', 'inst1_2', 'inst2_1', 'inst2_2', 'inst3_1', 'inst3_2', 'annotated']
    
    skipped = 0
    to_be_annotated_skip = 0
    files_rules = create_files_rules(rules)
    
    samples_rows = []
    
    for path, file_rules in tqdm(files_rules.items()):
        with open(os.path.join('../data/chirps_resource', path), 'rb') as rf:
            rules_inst = pickle.load(rf)
        for rule in file_rules:
            r1, r2 = rule
            data = rules_inst[tuple(sorted(rule))]
            tweets_data = tweets_tuple(data)
            if len(tweets_data) < 3:
                if '_'.join(rule) + '.pk' in neg_rules:
                    to_be_annotated_skip += 1
                skipped += 1
                continue
            samples = []
            org_rules = []
            for s in sample(tweets_data, 3):
                for tweet in s:
                    samples.append(tweet[1].replace('{a0}', "<font color='#1b9e77'>%s</font>"%tweet[0]).
                                   replace('{a1}', "<font color='#7570b3'>%s</font>"%tweet[2]).
                                   replace(clean_rule(tweet[1]), "<u>%s</u>"%clean_rule(tweet[1])))
                    org_rules.append(tweet[1])
            row = [r1, r2] + org_rules + samples + [str(is_annotated[rule])]
            samples_rows.append(row)
    
    logger.info('created {} samples'.format(len(samples_rows) - 1))
    logger.info('total skipping: {}, {}'.format(skipped, to_be_annotated_skip))
    
    shuffle(samples_rows)
    rows_chuks = chunks(samples_rows, 50)
    
    for i, rows_chunk in enumerate(rows_chuks):
        with codecs.open(os.path.join(args.out, '{}_samples{}.csv'.format(args.data_split, i+1)), 'w', 'utf-8') as f_out:
            logging.info('start writing to {}'.format(f_out.name))
            print('"' + '","'.join(title) + '"', file=f_out)
            for row in rows_chunk:
                print('"' + '","'.join([r.replace('"', "'") for r in row]) + '"', file=f_out)
          
    
    logger.info('created samples for annotation task')
    

def remove_dup(split):
    """
    To avoid double annotations of rules that appears in more than one split, we removed from test rules that are also in dev and train and from dev what is also in train, from eval_rules - what in all other splits.
    """
    rules = set(os.listdir(os.path.join(data_dirs[split], 'good')))
    logger.info('number of rules in {}: {}'.format(split, len(rules)))
    if split != 'train':
        train_rules = get_split_rules('train')
        if split == 'dev':
            rules = rules - train_rules
        elif split == 'test' or split == 'eval':
            dev_rules = get_split_rules('dev')
            rules = rules - train_rules
            rules = rules - dev_rules
            if split == 'eval':
                test_rules = get_split_rules('test')
                rules = rules - test_rules

    logger.info('number of rules in {} after removing dup from other splits: {}'.format(split, len(rules)))
    return list(rules)
             

def get_split_rules(split):
    rules = [os.listdir(os.path.join(data_dirs[split], d)) for d in dirs]
    rules = [rule for rules_dir in rules for rule in rules_dir]
    return set(rules)
    
    
def tweets_tuple(rule_data):
    tupels = set()
    for pair in rule_data:
        t1, t2 = pair
        args = [t1['arg0'], t1['arg1'], t2['arg0'], t2['arg1']]
        args = [arg.lower() for arg in args]
        if 'the' in args:
            continue
        tup1, tup2 = (t1['arg0'], t1['rule'], t1['arg1']), (t2['arg0'], t2['rule'], t2['arg1'])
        if (tup2, tup1) not in tupels:
            tupels.add((tup1, tup2))
    return tupels
   

def create_files_rules(rules):
    files_rules = defaultdict(list)
    for rule_p in rules:
        #r1, r2 = rule_p.split('.')[0].split('_')
        rule = extract_rule(rule_p)
        rule_path  = rules_instances[rule]
        files_rules[rule_path].append(rule)
    return files_rules


def get_rules_instances(r1, r2):
    rule = tuple(sorted([r1, r2]))
    rule_path  = rules_instances[rule]
    with open(os.path.join('../data/chirps_resource', rule_path), 'rb') as rf:
        rule_inst = pickle.load(rf)
    return rule_inst[rule]


def clean_rule(rule):
    rule = rule.replace('{a0}', '').replace('{a1}', '').strip()
    rule = re.sub(' +', ' ', rule)
    return rule
    
def extract_rule(rule_file):
    return tuple(sorted(rule_file.split('.')[0].split('_')))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
if __name__ == '__main__':
    main()
