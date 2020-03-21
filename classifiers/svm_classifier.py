from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_validate, ShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from pprint import pprint
import random
import numpy as np
import pickle
import json

from random_forest_params import best_params_randomize_search
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append('../rules_ranking/')
from rank_utils import AP

train_file = 'datasets/Kian/train'
dev_file = 'datasets/Kian/dev'

indexs_file = 'datasets/basic/split_indexs'

logger.info('loading train data from {}'.format(train_file))
with open(train_file, 'rb') as f:
    train = pickle.load(f)

with open(dev_file, 'rb') as f:
    dev = pickle.load(f)

t_rules, t_X, t_y = zip(*train)
t_X = np.array(t_X)
t_y = np.array(t_y)
t_rules = np.array(t_rules)

with open('features', 'r') as f:
    features = json.load(f)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

random_seed = 46

score_list = ['accuracy', 'precision', 'recall', 'f1', 'AP']
def random_forest_cv(data):
    rules, x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)

    best_params = best_params_randomize_search
    scores = defaultdict(lambda: np.array([]))
    print('random forest params:')
    pprint(best_params)
    for train_inx, test_inx in tqdm(cv.split(x)):
        train_X, train_y = x[train_inx], y[train_inx]
        test_X, test_y = x[test_inx], y[test_inx]
        clf = RandomForestClassifier(bootstrap=best_params['bootstrap'],
                                     max_depth=best_params['max_depth'],
                                     max_features=best_params['max_features'],
                                     min_samples_leaf=best_params['min_samples_leaf'],
                                     min_samples_split=best_params['min_samples_split'],
                                     n_estimators=best_params['n_estimators'],
                                     random_state=0)
        clf.fit(train_X, train_y)
        calculate_scores(train_X, train_y, clf, 'train', scores)
        calculate_scores(test_X, test_y, clf, 'test', scores)

    for key in scores.keys():
        print_scores(scores, key)
    return scores, clf


def random_forest(train, dev):
    rules, x, y = zip(*train)
    rules_dev, X_dev, y_dev = zip(*dev)
    best_params = best_params_randomize_search
    scores = defaultdict(lambda: np.array([]))
    print('random forest params:')
    pprint(best_params)
    clf = RandomForestClassifier(bootstrap=best_params['bootstrap'],
                                 max_depth=best_params['max_depth'],
                                 max_features=best_params['max_features'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 min_samples_split=best_params['min_samples_split'],
                                 n_estimators=best_params['n_estimators'],
                                 random_state=0)
    clf.fit(x, y)
    train_scores = calculate_score(x, y, clf)
    dev_scores = calculate_score(X_dev, y_dev, clf)
    calculate_AP(dev, clf, 'ranks/dev_random_forest.txt')
    calculate_AP(train, clf, 'ranks/train_random_forest.txt')
    for score, train_score in zip(score_list, train_scores):
        print('train {}: {}'.format(score, train_score))
    for score, dev_score in zip(score_list, dev_scores):
        print('dev {}: {}'.format(score, dev_score))

    return clf


def rf_hyperparameter_search(train, randomize, cv):
    rules, x, y = zip(*train)
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 10, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    if randomize:
        params_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=cv, verbose=2,
                                           random_state=42, n_jobs=-1)
    else:
        params_search = GridSearchCV(estimator=rf, param_grid=random_grid, cv=cv, n_jobs=-1, verbose=2)
    params_search.fit(x, y)
    pprint(params_search.best_params_)


def classifier_ablation(features_abl, save_to=None):
    inxs_remove = features_inx(features_abl)
    X_abl = remove_features(t_X, inxs_remove)
    clf_abl = train_clf(X_abl)
    if save_to:
        path = 'svm/ablations/' + save_to
        with open(path, 'wb') as f:
            pickle.dump(clf_abl)
        print('classifier without {} saved to {}'.format(features_abl, path))
    else:
        print('the classifier wasn\'t saved since the file path wasn\'t specified')
    return clf_abl


def features_inx(features_list):
    inxs = []
    for i, feat in enumerate(features):
        for f in features_list:
            if f in feat:
                inxs.append(i)
    return inxs


def remove_features(t_X, inxs):
    #  inx = features_inx(features_remove)
    new_X = np.array(t_X)
    new_X = np.delete(new_X, inxs, axis=1)
    if new_X.shape == np.array(t_X).shape:
        print('error at remove_features')
    return new_X


def train_clf(X):
    clf = svm.SVC(kernel='linear', C=0.1, random_state=random_seed)
    clf.fit(X, t_y)
    pred = clf.predict(X)
    print('accuracy: {}'.format(accuracy_score(t_y, pred)))
    prf = precision_recall_fscore_support(t_y, pred)
    print('precision: {}\trecsll: {}\tf1: {}'.format(prf[0][1], prf[1][1], prf[2][1]))
    return clf


def create_tsne_graph(X, y, save_to):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    df = pd.DataFrame()
    df['y'] = y
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig(save_to)


def train_svm(train, test, pos_weight, save_to=None, rules_proba_file=None):
    c = 0.1
    kernel = 'linear'
    # train
    logger.info('start training svm:c-{}, kernel-{}, pos_weight-{}'.format(c, kernel, pos_weight))

    rules, X, y = zip(*train)
    clf = svm.SVC(kernel=kernel, C=c, probability=True, class_weight={0: 1, 1: pos_weight}, random_state=random_seed)
    clf.fit(X, y)
    pred_X = clf.predict(X)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred_X, labels=[1, 0])
    logger.info('finish training')
    print('train accuracy:\t{}'.format(clf.score(X, y)))
    print('train precision:\t{}'.format(precision))
    print('train recall:\t{}'.format(recall))
    print('train f1:\t{}'.format(f1))
    print('train AP:\t{}'.format(calculate_AP(train, clf, 'ranks/train_svm_weight_{}.txt'.format(pos_weight))))
    print('train support:\t{}'.format(support))

    # test
    _, X_test, y_test = zip(*test)
    pred_X_test = clf.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, pred_X_test, labels=[1, 0])
    print('test accuracy:\t{}'.format(clf.score(X_test, y_test)))
    print('test precision:\t{}'.format(precision))
    print('test recall:\t{}'.format(recall))
    print('test f1:\t{}'.format(f1))
    print('test AP:\t{}'.format(calculate_AP(test, clf, 'ranks/test_svm_weight_{}.txt'.format(pos_weight))))
    print('test support:\t{}'.format(support))

    if save_to:
        with open(save_to, 'wb') as clf_f:
            pickle.dump(clf, clf_f)
        logger.info('clf saved to {}'.format(save_to))

    rules_proba = calculate_proba(train, clf)
    rules_proba.update(calculate_proba(test, clf))

    if rules_proba_file:
        with open(rules_proba_file, 'wb') as rules_f:
            pickle.dump(rules_proba, rules_f)
        logger.info('clf saved to {}'.format(save_to))
    return clf


def calculate_proba(data, clf):
    rules, X, y = zip(*data)
    proba = clf.predict_proba(X)[:, np.where(clf.classes_ == 1)[0][0]]
    rules_dict = {r: p for r, p in zip(rules, proba)}
    return rules_dict


def remove_samples(split, num):
    random.seed(2)
    pos = [s for s in split if s[2] == 1]
    neg = random.sample([s for s in train if s[2] == 0], num)
    balanced_split = pos + neg
    logger.info('pos:{}, neg:{}, all:{}'.format(len(pos), len(neg), len(balanced_split)))
    random.shuffle(balanced_split)
    return balanced_split


def cv_svm_SMOTE(train, c, kf):
    rules, X, y = zip(*train)
    kernel = 'linear'
    X = np.array(X)
    y = np.array(y)
    scores = defaultdict(lambda: np.array([]))
    for train_inx, test_inx in kf.split(X):
        train_X, train_y = X[train_inx], y[train_inx]
        test_X, test_y = X[test_inx], y[test_inx]
        sm = SMOTE(random_state=2)
        X, Y = sm.fit_sample(train_X, train_y)
        logger.info('1:{}, 0:{}'.format(sum(Y), len(Y) - sum(Y)))
        clf = svm.SVC(kernel=kernel, C=c, probability=True, random_state=random_seed)
        calculate_scores(train_X, train_y, clf, 'train', scores)
        calculate_scores(test_X, test_y, clf, 'train', scores)

        # clf.fit(X, Y)
        # train_pred = clf.predict(train_X)
        # train_prf = precision_recall_fscore_support(train_y, train_pred, labels=[1, 0])
        # train_accuracy = accuracy_score(train_y, train_pred)
        # scores['train_accuracy'] = np.append(scores['train_accuracy'], train_accuracy)
        # scores['train_precision'] = np.append(scores['train_precision'], train_prf[0][0])
        # scores['train_recall'] = np.append(scores['train_recall'], train_prf[1][0])
        # scores['train_f1'] = np.append(scores['train_f1'], train_prf[2][0])
        # scores['train_AP'] = np.append(scores['train_AP'], calculate_AP_s(train_X, train_y, clf))
        # test_pred = clf.predict(test_X)
        # test_prf = precision_recall_fscore_support(test_y, test_pred, labels=[1, 0])
        # test_accuracy = accuracy_score(test_y, test_pred)
        # scores['test_accuracy'] = np.append(scores['test_accuracy'], test_accuracy)
        # scores['test_precision'] = np.append(scores['test_precision'], test_prf[0][0])
        # scores['test_recall'] = np.append(scores['test_recall'], test_prf[1][0])
        # scores['test_f1'] = np.append(scores['test_f1'], test_prf[2][0])
        # scores['test_AP'] = np.append(scores['test_AP'], calculate_AP_s(test_X, test_y, clf))

    for score in scores.keys():
        print_scores(scores, score)


def cv_svm(X, y, c, kf, calc_AP=False, save_rank=None, class_weight=5):
    scores = defaultdict(lambda: np.array([]))
    kernel = 'linear'
    logger.info('{} kernel'.format(kernel))
    if type(kf) is int:
        clf = svm.SVC(kernel=kernel, C=c, probability=True, random_state=random_seed)
        scores = cross_validate(clf, X, y, cv=kf, scoring=('accuracy', 'precision', 'recall', 'f1'),
                                return_train_score=True)
        for key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'train_accuracy', 'train_precision',
                    'train_recall', 'train_f1']:
            print_scores(scores, 'test_accuracy')
        calc_AP = False
    else:
        for train_inx, test_inx in tqdm(kf.split(X)):
            train_X, train_y = X[train_inx], y[train_inx]
            test_X, test_y = X[test_inx], y[test_inx]
            clf = svm.SVC(kernel=kernel, C=c, probability=True, class_weight={0: 1, 1: class_weight},
                          random_state=random_seed)
            clf.fit(train_X, train_y)
            # calculate_scores(train_X, train_y, clf, 'train', scores)
            # calculate_scores(test_X, test_y, clf, 'dev', scores)
            train_pred = clf.predict(train_X)
            train_prf = precision_recall_fscore_support(train_y, train_pred, labels=[1, 0])
            train_accuracy = accuracy_score(train_y, train_pred)
            scores['train_accuracy'] = np.append(scores['train_accuracy'], train_accuracy)
            scores['train_precision'] = np.append(scores['train_precision'], train_prf[0][0])
            scores['train_recall'] = np.append(scores['train_recall'], train_prf[1][0])
            scores['train_f1'] = np.append(scores['train_f1'], train_prf[2][0])
            scores['train_AP'] = np.append(scores['train_AP'], calculate_AP_s(train_X, train_y, clf))
            test_pred = clf.predict(test_X)
            test_prf = precision_recall_fscore_support(test_y, test_pred, labels=[1, 0])
            test_accuracy = accuracy_score(test_y, test_pred)
            scores['test_accuracy'] = np.append(scores['test_accuracy'], test_accuracy)
            scores['test_precision'] = np.append(scores['test_precision'], test_prf[0][0])
            scores['test_recall'] = np.append(scores['test_recall'], test_prf[1][0])
            scores['test_f1'] = np.append(scores['test_f1'], test_prf[2][0])
            scores['test_AP'] = np.append(scores['test_AP'], calculate_AP_s(test_X, test_y, clf))

        logger.info('Trained succssefully')
        for score in scores.keys():
            print_scores(scores, score)

        if calc_AP:
            test_rank = calculate_AP(X, c, clf)
        if save_rank:
            text_rank = ['\t'.join([r[2], str(r[1]), str(r[0])]) for r in test_rank]
            print(text_rank[:2])
            with open(save_rank, 'w') as f:
                f.write('\n'.join(text_rank))
    return scores


def calculate_scores(X, y, clf, split, scores):
    accuracy, precision, recall, f1, AP = calculate_score(X, y, clf)
    scores['{}_accuracy'.format(split)] = np.append(scores['{}_accuracy'.format(split)], accuracy)
    scores['{}_precision'.format(split)] = np.append(scores['{}_precision'.format(split)], precision)
    scores['{}_recall'.format(split)] = np.append(scores['{}_recall'.format(split)], recall)
    scores['{}_f1'.format(split)] = np.append(scores['{}_f1'.format(split)], f1)
    scores['{}_AP'.format(split)] = np.append(scores['{}_AP'.format(split)], AP)


def calculate_score(X, y, clf):
    pred = clf.predict(X)
    prf = precision_recall_fscore_support(y, pred, labels=[1, 0])
    accuracy = accuracy_score(y, pred)
    AP = calculate_AP_s(X, y, clf)
    return accuracy, prf[0][0], prf[1][0], prf[2][0], AP


def calculate_AP_s(X, y, clf):
    proba = clf.predict_proba(X)[:, np.where(clf.classes_ == 1)[0][0]]
    test_rank = sorted(zip(proba, y), key=lambda r: r[0], reverse=True)
    label_rank = list(zip(*test_rank))[1]
    AP_score = AP(label_rank)
    return AP_score


def calculate_AP(data, clf, save_to):
    rules, X, y = zip(*data)
    proba = clf.predict_proba(X)[:, np.where(clf.classes_ == 1)[0][0]]
    test_rank = sorted(zip(proba, y, rules), key=lambda r: r[0], reverse=True)
    label_rank = list(zip(*test_rank))[1]
    AP_score = AP(label_rank)
    save_clf_rank(test_rank, save_to.format(AP_score))
    return AP_score


def save_clf_rank(data, save_to):
    proba, y, rules = zip(*data)
    proba = list(map(lambda x: str(x), proba))
    y = list(map(lambda x: str(x), y))
    lines = map(lambda d: '\t'.join(d), zip(rules, proba, y))
    with open(save_to, 'w') as f:
        f.write('\n'.join(lines))


def print_scores(scores, key): print("%s: %0.2f (+/- %0.2f)" % (key, scores[key].mean(), scores[key].std() * 2))


def baseline_cv(train, kf):
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

    # for till here
    print(scores)
    for score_type, l in scores.items():
        scores[score_type] = np.array(l)
        print_scores(scores, score_type)

    return scores


def baseline(train, test):
    rules_train, X_train, y_train = zip(*train)
    rules_test, X_test, y_test = zip(*test)

    train_split = list(zip(X_train, y_train, rules_train))
    test_split = list(zip(X_test, y_test, rules_test))

    s_train = sorted(train_split, key=lambda x: x[0][4], reverse=True)
    s_X, s_y, s_rules = zip(*s_train)
    s_AP = AP(s_y)
    print('train AP: {}'.format(s_AP))
    save_baseline_rank(s_train, 'ranks/baseline_train_{}.txt'.format(s_AP))
    s_test = sorted(test_split, key=lambda x: x[0][4], reverse=True)
    s_X, s_y, s_rules = zip(*s_test)
    s_AP = AP(s_y)
    print('test AP: {}'.format(s_AP))

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
    print('threshold: {}'.format(ts))

    # print('precision: {}, recall: {}, f1: {}, accuracy: {}'.format
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


def f1_ts(i):
    pred = np.concatenate([np.ones(i), np.zeros(len(s_X) - i)], axis=None)
    try:
        prf = precision_recall_fscore_support(s_y, pred, labels=[1, 0])
    except:
        pass
    return (prf[0][0], prf[1][0], prf[2][0])


def features_ablation(num=len(features)):
    feats = features if not num else features[:num]
    for i, feat in enumerate(feats):
        new_X = remove_features(t_X, i)
        print('training without {} ({}/{})'.format(feat, i + 1, num))
        cv_svm(new_X, t_y, 0.1, 3)
        print('...')
        print('...')


def save_baseline_rank(data, path):
    X, y, rules = zip(*data)
    y = list(map(lambda x: str(x), y))
    X = list(map(lambda x: str(x[4]), X))
    lines = map(lambda d: '\t'.join(d), zip(rules, X, y))
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    f.close()


def sort_and_AP(X, inx, y):
    sorted_x = sorted(zip(X, y), key=lambda k: k[0][inx], reverse=True)
    ys = list(zip(*sorted_x))[1]
    return AP(ys)


def sort_by(X, inx, y, cv):
    APs = {'train': np.array([]),
           'test': np.array([])}
    for train_inx, test_inx in cv.split(X):
        train_X, train_y = X[train_inx], y[train_inx]
        test_X, test_y = X[test_inx], y[test_inx]
        APs['train'] = np.append(APs['train'], sort_and_AP(train_X, isc.nx, train_y))
        APs['test'] = np.append(APs['test'], sort_and_AP(test_X, inx, test_y))
    print_scores(APs, 'train')
    print_scores(APs, 'test')
    return APs


if __name__ == '__main__':
    with open('datasets/Kian/train_an', 'rb') as f:
        train_an = pickle.load(f)
    baseline_cv(train, cv)
