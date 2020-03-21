import tweets_clique as clq
import pickle
import os
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

directories = ['/home/nlp/yehudit/projects/coreferrability/data/tweets/train/bad_events_added/',
               '/home/nlp/yehudit/projects/coreferrability/data/tweets/dev/bad_events_added/']              
"""
              ['/home/nlp/yehudit/projects/coreferrability/data/tweets/train/bad_events_all/',
               '/home/nlp/yehudit/projects/coreferrability/data/tweets/train/good_events_all/',
               '/home/nlp/yehudit/projects/coreferrability/data/tweets/dev/bad_events_all/',
               '/home/nlp/yehudit/projects/coreferrability/data/tweets/dev/good_events_all/']
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The data pickle files directory")

    global args
    args = parser.parse_args()
    logger.info("data dir:\t{}".format(args.data_dir))

    logger.info('start calculating in_clique feature')
    
    dir_clique(args.data_dir)


def in_clique(id1, id2):
    return clq.is_in_clique(id1, id2)
    

def rule_clique(path):
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            logger.error('problen in data as {}'.format(f))
            raise
    for pair in data:
        tweet1, tweet2 = pair['tweets']
        id1, id2 = tweet1['id'], tweet2['id']
        pair['features']['in_clique'] = in_clique(id1, id2)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

        
def dir_clique(data_dir):
    logger.info('start calculating for {}'.format(data_dir))
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    logger.info('# of files:\t{}'.format(len(files)))
    for f in tqdm(files):
        rule_clique(f)
    logger.info('end calculation for {}'.format(data_dir))

        

if __name__ == '__main__':
    main()