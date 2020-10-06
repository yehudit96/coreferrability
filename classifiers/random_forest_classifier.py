from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_validate, ShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import random
import numpy as np
import pickle
import json
from scipy.spatial.distance import cosine

from random_forest_params import best_params_randomize_search
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append('../rules_ranking/')
from rank_utils import AP


with open('features', 'r') as f:
    features = json.load(f)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

random_seed = 46

score_list = ['accuracy', 'precision', 'recall', 'f1', 'AP']


def random_forest_cv(data):
    '''
    Train and test a random-forest classifier, with cross-validation with 5 splits and train-test ratio of 80-20
    
    Parametes:
        data (list): the data for the training and testing, the data is a list of tuples (rule, x, y)
    
    Return:
        scores (list): list of scores on train and test sets (for each fold)
        clf (RandomForestClassifier): the last trained classifier
    '''
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
    return scores


def random_forest(train, dev, print_params = False):
    """
    train and test random forest classifier using parametesr from hyperparameters search
    Parameters:
        train (list): the train data
        dev (list): the dev data
        parint_params (bool): flag to print the random-forest parameters 
    Return:
        trained random-forest classifier
    """
    # extract the data to separate list for rules name, x - feature vector and y - rules labels 
    rules, x, y = zip(*train)
    rules_dev, X_dev, y_dev = zip(*dev)
    best_params = best_params_randomize_search
    scores = defaultdict(lambda: np.array([]))
    if print_params:
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
    """
    search for the best hyperparameters
    Parameters:
        train (list): the train data
        randomize (bool): flag weather to do randomize or grid search
        cv: cross validation split
    Return:
        best parameters
    """
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
    return params_search.best_params_


def calculate_scores(X, y, clf, split, scores):
    """
    calculate the scores of for a given classifier and add to a scores dictionary (for cross validation)
    Parameters:
        X: list of feature vectors
        y: list of labels
        clf: clssifier
        split: split name
        scores: doctopnary of the scores
    Return:
        Nothing (the scores saved to the given scores dictionary)
    """
    accuracy, precision, recall, f1, AP = calculate_score(X, y, clf)
    scores['{}_accuracy'.format(split)] = np.append(scores['{}_accuracy'.format(split)], accuracy)
    scores['{}_precision'.format(split)] = np.append(scores['{}_precision'.format(split)], precision)
    scores['{}_recall'.format(split)] = np.append(scores['{}_recall'.format(split)], recall)
    scores['{}_f1'.format(split)] = np.append(scores['{}_f1'.format(split)], f1)
    scores['{}_AP'.format(split)] = np.append(scores['{}_AP'.format(split)], AP)


def calculate_score(X, y, clf):
    """
    clalculate scores - accuracy, precision, recall, f1 and AP for a given classifier
    Parameters:
        X: list of feature vectors
        y: list of labels
        clf: clssifier
    Return:
        accuracy, precision, recall, f1, AP scores
    """
    pred = clf.predict(X)
    prf = precision_recall_fscore_support(y, pred, labels=[1, 0])
    accuracy = accuracy_score(y, pred)
    AP = calculate_AP_s(X, y, clf)
    return accuracy, prf[0][0], prf[1][0], prf[2][0], AP


def calculate_AP_s(X, y, clf):
    """
    calculate the AP by the classifier proba
    Parameters:
        X: list of feature vectors
        y: list of labels
        clf: clssifier
    Return:
        the AP score
    """
    proba = clf.predict_proba(X)[:, np.where(clf.classes_ == 1)[0][0]]
    test_rank = sorted(zip(proba, y), key=lambda r: r[0], reverse=True)
    label_rank = list(zip(*test_rank))[1]
    AP_score = AP(label_rank)
    return AP_score


def calculate_AP(data, clf, save_to):
    """
    calculate the AP by the classifier proba and saves the ranked rules with the label and the score to save_to path
    Parameters:
        data: list of data tupels (rule name, X, y)
        clf: clssifier
        save_to: the path where to save the ranks
    Return:
        the AP score
    """
    rules, X, y = zip(*data)
    proba = clf.predict_proba(X)[:, np.where(clf.classes_ == 1)[0][0]]
    test_rank = sorted(zip(proba, y, rules), key=lambda r: r[0], reverse=True)
    label_rank = list(zip(*test_rank))[1]
    AP_score = AP(label_rank)
    save_clf_rank(test_rank, save_to.format(AP_score))
    return AP_score


def save_clf_rank(data, save_to):
    """
    save the ranking data
    Parameters:
        data: list of data tupels (score, y, rule name)
        save_to: the path where to save the ranks
    Return:
    """
    proba, y, rules = zip(*data)
    proba = list(map(lambda x: str(x), proba))
    y = list(map(lambda x: str(x), y))
    lines = map(lambda d: '\t'.join(d), zip(rules, proba, y))
    with open(save_to, 'w') as f:
        f.write('\n'.join(lines))


def print_scores(scores, key): print("%s: %0.2f (+/- %0.2f)" % (key, scores[key].mean(), scores[key].std() * 2))


if __name__ == '__main__':
    with open('datasets/Kian/train_an', 'rb') as f:
        train_an = pickle.load(f)
    baseline_cv(train, cv)
