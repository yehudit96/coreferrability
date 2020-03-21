import random
import itertools
import functools
import pickle
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from svm_classifier import AP, cv

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--n", default=None, type=int, required=True, help="number of training samples")
    parser.add_argument("--print_p", default=False, type=bool, required=False, help="number of training samples")
    parser.add_argument("--clf_type", default=False, type=str, required=True, help=" classifier type - rf (random forest) | svm")
    parser.add_argument("--save_to", default=False, type=str, required=True, help=" file path to save the ranker")
    
    
    args = parser.parse_args()
    
    SAVE_TO = args.save_to
    
    with open('datasets/Kian/train_an', 'rb') as f:
        train = pickle.load(f)
    
    with open('datasets/Kian/dev_an', 'rb') as f:
        dev = pickle.load(f)
    
    _, X, y = zip(*train)
    _, dev_X, dev_y = zip(*dev)
    X_train, y_train, = np.array(X), np.array(y)
    X_dev, y_dev = np.array(dev_X), np.array(dev_y)
    
    logger.info('creating training instances')    
    Xp, yp = create_pair_instances(X_train, y_train)

    logger.info('creating dev instances')    
    Xp_dev, yp_dev = create_pair_instances(X_dev, y_dev)

    #samples = random.sample(list(zip(Xp, yp)), args.n)
    #Xp, yp = zip(*samples)
    logger.info('{} pairs'.format(len(Xp)))
    #Xp, Xp_test, yp, yp_test = train_test_split(Xp, yp, test_size=1-(float(args.n)/len(Xp)), random_state=42)
    #logger.info('created {} training instances, -1:{}, 1:{}'.format(len(Xp), sum(1 for y in yp if y==-1), sum(1 for y in yp if y==1)))
    #logger.info('created {} test instances, -1:{}, 1:{}'.format(len(Xp_test), sum(1 for y in yp_test if y==-1), sum(1 for y in yp_test if y==1)))
    
    if args.clf_type == 'svm':
        logger.info('start training svm ranker')
        ranker = svm.SVC(kernel='linear', C=0.1, verbose=args.print_p)
    else:
        logger.info('start training random forest ranker')
        best_params = {'bootstrap': True,
                                        'max_depth': 8,
                                        'max_features': 'auto',
                                        'min_samples_leaf': 1,
                                        'min_samples_split': 10,
                                        'n_estimators': 157}
        ranker = RandomForestClassifier(bootstrap = best_params['bootstrap'],
                                        max_depth=best_params['max_depth'], 
                                        max_features = best_params['max_features'],
                                        min_samples_leaf = best_params['min_samples_leaf'], 
                                        min_samples_split=best_params['min_samples_split'],
                                        n_estimators=best_params['n_estimators'],
                                        random_state=0)

    ranker.fit(Xp, yp)
    logger.info('trained svm ranker')

    logger.info('svm ranker train accuracy: {}'.format(ranker.score(Xp, yp)))
    logger.info('svm ranker test accuracy: {}'.format(ranker.score(Xp_dev, yp_dev)))

    with open(SAVE_TO, 'wb') as f:
        pickle.dump(ranker, f)
    
    logger.info('ranker saved to {}'.format(SAVE_TO))

    logger.info('re-ranking rules by trained ranker')
    
    def compare_rules(r1, r2):
        return ranker.predict((pair_to_vec(r1[0], r2[0])).reshape(1, -1))
        
    sorted_rules_train = sorted(zip(X_train, y_train), key=functools.cmp_to_key(compare_rules), reverse=True)
    print('sorted rules train AP:{}'.format(AP(list(zip(*sorted_rules_train))[1])))

    sorted_rules_dev = sorted(zip(X_dev, y_dev), key=functools.cmp_to_key(compare_rules), reverse=True)
    print('sorted rules dev AP:{}'.format(AP(list(zip(*sorted_rules_dev))[1])))
    
def pair_to_vec(x1, x2):
    return x1 - x2 #np.concatenate((x1, x2))
    
    
def create_pair_instances(X, y):
    indices = list(range(len(X)))
    random.shuffle(indices)

    Xp, yp = [], []
    for i1, i2 in itertools.combinations(indices, 2):
        if y[i1] == y[i2]:
            continue
        Xp.append(pair_to_vec(X[i1], X[i2]))
        yp.append(np.sign(y[i1] - y[i2]))
    return Xp, yp

if __name__ == '__main__':
    main()
