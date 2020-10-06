from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
import gensim.downloader as api
model = api.load('glove-wiki-gigaword-300')


import sys
sys.path.append('../rules_ranking/')
from rank_utils import AP
from data_loader import train, test

"""
this scrip computes the baseline algorithm scores
"""

def baseline_cv(train, kf):
    """
    train by baseline algorithm - finding the threshold which yelding the best f1 scores (sorting according to chirps score).
    Parameters:
        train: train data
        kf: cross-validation splitter
    Retutn:
        split scores
    """
    scores = defaultdict(list)
    rules, X, y = zip(*train)
    rules = np.array(rules)
    X = np.array(X)
    y = np.array(y)
    for train_inx, test_inx in tqdm(kf.split(X)):
        rules_train, X_train, y_train = rules[train_inx], X[train_inx], y[train_inx]
        rules_test, X_test, y_test = rules[test_inx], X[test_inx], y[test_inx]

        train_split = list(zip(X_train, y_train, rules_train))
        test_split = list(zip(X_test, y_test, rules_test))

        s_train = sorted(train_split, key=lambda x: x[0][4], reverse=True)
        s_X, s_y, s_rules = zip(*s_train)
        s_AP = AP(s_y)
        scores['train_APs'].append(s_AP)

        save_baseline_rank(s_train, 'ranks/baseline_train_{}.txt'.format(s_AP))
        s_test = sorted(test_split, key=lambda x: x[0][4], reverse=True)
        s_X, s_y, s_rules = zip(*s_test)
        s_AP = AP(s_y)
        scores['test APs'].append(s_AP)

        save_baseline_rank(s_test, 'ranks/baseline_test_{}.txt'.format(s_AP))
        train_chirps = np.array([t[0][4] for t in train_split])
        train_y = np.array([t[1] for t in train_split])

        test_chirps = np.array([t[0][4] for t in test_split])
        test_y = np.array([t[1] for t in test_split])

        results = []
        for i, ts in enumerate(train_chirps):
            pred = np.where(train_chirps > ts, 1, 0)
            accuracy = accuracy_score(train_y, pred)
            prf = precision_recall_fscore_support(train_y, pred, labels=[1, 0])
            ts_scores = prf[0][0], prf[1][0], prf[2][0], accuracy
            results.append((i, ts, ts_scores))

        # max f1 result
        max_i, ts, train_scores = max(results, key=lambda r: r[2][2])
        print(train_scores)
        scores['thresholds'].append(ts)

        scores['train_precision'].append(train_scores[0])
        scores['train_recall'].append(train_scores[1])
        scores['train_f1'].append(train_scores[2])
        scores['train_accuracy'].append(train_scores[3])

        test_pred = np.where(test_chirps > ts, 1, 0)
        prf = precision_recall_fscore_support(test_y, test_pred, labels=[1, 0])
        test_scores = prf[0][0], prf[1][0], prf[2][0]
        scores['test_presicion'].append(test_scores[0])
        scores['test_recall'].append(test_scores[1])
        scores['test_f1'].append(test_scores[2])
        scores['test_accuracy'].append(accuracy_score(test_y, test_pred))

    print(scores)
    for score_type, l in scores.items():
        scores[score_type] = np.array(l)
        print_scores(scores, score_type)

    return scores
    

def baseline(train, test, score_func):
    """
    train by baseline algorithm - finding the threshold which yelding the best accuracy scores (sorting according given score_func).
    Parameters:
        train: train data
        test: test data
        score_func: the scoring function for a given rule
    Retutn:
    """
    rules_train, X_train, y_train = zip(*train)
    rules_test, X_test, y_test = zip(*test)

    train_split = list(zip(X_train, y_train, rules_train))
    test_split = list(zip(X_test, y_test, rules_test))

    s_train = sorted(train_split, key=lambda x:score_func(x), reverse=True)
    s_X, s_y, s_rules = zip(*s_train)
    s_AP = AP(s_y)
    print('train AP: {}'.format(s_AP))
    save_baseline_rank(s_train, 'ranks/baseline_train_{}.txt'.format(s_AP))
    s_test = sorted(test_split, key=lambda x:score_func(x),  reverse=True)
    s_X, s_y, s_rules = zip(*s_test)
    s_AP = AP(s_y)
    print('test AP: {}'.format(s_AP))

    save_baseline_rank(s_test, 'ranks/baseline_test_{}.txt'.format(s_AP))

    train_chirps = np.array([score_func(t) for t in train_split])
    train_y = np.array([t[1] for t in train_split])

    test_chirps = np.array([score_func(t) for t in test_split])
    test_y = np.array([t[1] for t in test_split])

    results = []
    for i, ts in enumerate(train_chirps):
        pred = np.where(train_chirps > ts, 1, 0)
        accuracy = accuracy_score(train_y, pred)
        prf = precision_recall_fscore_support(train_y, pred, labels=[1, 0])
        ts_scores = prf[0][0], prf[1][0], prf[2][0], accuracy
        results.append((i, ts, ts_scores))

    # max accuracy result
    max_i, ts, train_scores = max(results, key=lambda r: r[2][3])
    print(train_scores)
    print('threshold: {}'.format(ts))

    print('train_precision: {}'.format(train_scores[0]))
    print('train_recall: {}'.format(train_scores[1]))
    print('train_f1: {}'.format(train_scores[2]))
    print('train_accuracy: {}'.format(train_scores[3]))

    test_pred = np.where(test_chirps > ts, 1, 0)
    prf = precision_recall_fscore_support(test_y, test_pred, labels=[1, 0])
    test_scores = prf[0][0], prf[1][0], prf[2][0]
    print('test_presicion: {}'.format(test_scores[0]))
    print('test_recall: {}'.format(test_scores[1]))
    print('test_f1: {}'.format(test_scores[2]))
    print('test_accuracy: {}'.format(accuracy_score(test_y, test_pred)))


def save_baseline_rank(data, path):
    """
    save the baseline rank to path
    Parameters:
        data: sorted data
        path: where to save the ranking
    Return:
    """
    X, y, rules = zip(*data)
    y = list(map(lambda x: str(x), y))
    X = list(map(lambda x: str(x[4]), X))
    lines = map(lambda d: '\t'.join(d), zip(rules, X, y))
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    f.close()


"""
Functions for baseline score_func
"""

def score_GloVe(x):
    r1, r2 = x[2].split('_')
    try:
        v1 = model.wv[r1] if len(r1.split()) == 1 else np.sum([model.wv[w] for w in r1.split()])
        v2 = model.wv[r2] if len(r2.split()) == 1 else np.sum([model.wv[w] for w in r2.split()])
    except:
        print('{}, {} not in GloVe dictionary'.format(r1, r2))
        return 0
    return 1 - cosine(v1, v2)


def score_chirps(x):
    return x[0][4]
    
if __name__ == '__main__':
    print('baseline by chirps score:')
    baseline(train, test, score_chirps)
    print('baseline by GloVe score:')
    baseline(train, test, score_GloVe)
    