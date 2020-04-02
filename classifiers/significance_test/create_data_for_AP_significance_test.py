import os
import sys
import random
import argparse
import _pickle as cPickle
from random import choices
from tqdm import tqdm
random.seed(1)

from svm_classifier import *
parser = argparse.ArgumentParser(description='Creating data for statistical significance tests')

parser.add_argument('--rules_path', type=str,
                    help=' The path to the rules test set')
parser.add_argument('--classifier', type=str,
                    help='  The path to the classifier')
parser.add_argument('--out_dir', type=str,
                    help='Output folder')

args = parser.parse_args()


def test_significance():
    """
    Runs the whole process of creating the results for statistical significance tests.
    """    
    with open(args.rules_path, 'rb') as f_rules:
        data = pickle.load(f_rules)
    with open(args.classifier, 'rb') as f_clf:
         clf = pickle.load(f_clf)
    rules_scores = {}
    rules, x, y = zip(*data)
    pred = clf.predict_proba(x)[:, np.where(clf.classes_ == 1)[0][0]]
    rules = list(zip(y, map(lambda a: a[4], x), pred))
    chirps_scores = []
    clf_scores = []
    for i in tqdm(range(1000)):
        selected = choices(rules, k=len(rules))
        chirps_sort = sorted(selected, key=lambda r:r[1], reverse=True)
        clf_sort = sorted(selected, key=lambda r:r[2], reverse=True)
        chirps_AP = AP(map(lambda r: r[0], chirps_sort))
        clf_AP = AP(map(lambda r: r[0], clf_sort))
        chirps_scores.append(str(chirps_AP))
        clf_scores.append(str(clf_AP))
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open(os.path.join(args.out_dir, 'a_scores.txt'), 'w') as f_a:
        f_a.write('\n'.join(chirps_scores))
    with open(os.path.join(args.out_dir, 'b_scores.txt'), 'w') as f_b:
        f_b.write('\n'.join(clf_scores))


def main():
    """
    This script runs the whole process of creating the results for statistical significance tests,
    which includes sampling of 1000 topics combinations, extracting the results of system A and system
    B for those combinations, running CoNLL scorer for each system in each topics combination
    and extracting the CoNLL results.
    :return:
    """
    test_significance()


if __name__ == '__main__':
    main()