import pickle
import os
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""
create pickled file of dictionary of svm score for rule
"""
vectors_files = ['train', 'dev', 'test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf", default=None, type=str, required=True, help=" clf path for rules scores computation")
    parser.add_argument("--vectors_path", default=None, type=str, required=True, help=" vectors files dir")
    parser.add_argument("--scores_path", default=None, type=str, required=True, help=" rules scores out file")
    args = parser.parse_args()
    
    with open(args.clf, 'rb') as f_clf:
        clf = pickle.load(f_clf)
    rules_scores = {}
    for split in vectors_files:
        logger.info('start predicting for {} split'.format(split))
        path = os.path.join(args.vectors_path, split)
        with open(path, 'rb') as f_vcr:
            data = pickle.load(f_vcr)
        vectors = [d[1] for d in data]
        rules = [d[0] for d in data]
        scores = clf.predict(vectors)
        split_scores = {r: s for r, s in zip(rules, scores)}
        rules_scores.update(split_scores)
        if split == 'train':
            logger.info('train average: {}'.format(sum(scores)/len(scores)))
    
    logger.info('created rule - score dictionary with {} rules'.format(len(rules_scores)))
    with open(args.scores_path, 'wb') as f_out:
        pickle.dump(rules_scores, f_out)
    logger.info('saved successfully to {}'.format(args.scores_path))


if __name__ == '__main__':
    main()