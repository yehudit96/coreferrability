import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import sys
rank_path = '../rules_ranking'
sys.path.append(rank_path)
from rank_utils import AP
from baseline_classifier import score_GloVe

random.seed(1)

skip = 15

def main():
    with open('datasets/Kian/eval_an', 'rb') as f:
        eval_rules = pickle.load(f)
    
    eval_rules = random.sample(eval_rules, 500)
    rules, x, y = zip(*eval_rules)
    with open('best_rf', 'rb') as f_clf:
        clf = pickle.load(f_clf)

    pred = clf.predict(x)
    pred_proba = clf.predict_proba(x)
    proba = list(zip(x,y,rules, pred_proba, pred))
    sort_proba = sorted(proba, key=lambda a:a[3][1], reverse=True)
    _, y, rules_model, _, pred = zip(*sort_proba)
    inx = list(map(int, np.linspace(0,len(y), 5)))
    bins = list(zip(inx[:-1], inx[1:])) 
    print('sort by classifier')
    for bin_s, bin_e in bins:
        print("number of positive rules between {} to {}".format(bin_s, bin_e))
        print(sum(y[bin_s:bin_e]))
    print('classifier AP: {}'.format(AP(y)))
    print_top_10(y, rules_model)
    
    
    sort_chirps = sorted(proba, key=lambda a:a[0][4], reverse=True)
    _, y, rules_chirps, _, pred = zip(*sort_chirps)
    print('sort by chirps rank')
    for bin_s, bin_e in bins:
        print("number of positive rules between {} to {}".format(bin_s, bin_e))
        print(sum(y[bin_s:bin_e]))
    print('chirps AP: {}'.format(AP(y)))
    print_top_10(y, rules_chirps)
    
    sort_GloVe = sorted(proba, key=lambda a:score_GloVe(a), reverse=True)
    _, y, rules_GloVe, _, pred = zip(*sort_GloVe)
    print('sort by GloVe rank')
    for bin_s, bin_e in bins:
        print("number of positive rules between {} to {}".format(bin_s, bin_e))
        print(sum(y[bin_s:bin_e]))
    print('GloVe AP: {}'.format(AP(y)))
    print_top_10(y, rules_GloVe)
    
    
def print_top_10(labels, rules):
    print('skip: {}'.format(skip))
    for i, label, rule in list(zip(range(1, len(labels)+1), labels, rules))[0::skip][:10]:
        print('{}: {}'.format(label, rule, i))


if __name__ == "__main__":
    main()
