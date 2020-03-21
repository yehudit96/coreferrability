import json
import pickle
from collections import defaultdict
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_text', type=str, required=True, help='Corpus text file')
    parser.add_argument('--topics_path', type=str, required=True, help='path where to save the topics list')
    args = parser.parse_args()    
    for arg, value in args._get_kwargs():
        logger.info('{}:\t{}'.format(arg, value))
    
def create_topics_list(corpus_text, topics_path):
    topics = defaultdict(list)
    
    logger.info('creating topics list..')
    #create topics dict, for each topic entery its doc_ids added 
    with open(corpus_text) as f:
        first = True
        for l in f:
            if l == '\n':
                first = True
            elif first:
                d = l.strip().split('\t')
                doc_id = d[0]
                topic = doc_id.split('_')[1]
                topics[topic].append(doc_id)
                first = False
                
    #check if every topic has exactly 2 doc_ids
    for t_id, topic in topics.items():
        assert len(topic) == 2, 'topic {} has {} documents'.format(t_id, len(topic))

    logger.info('created topics list successfuly')
    
    #save topics list to topics path
    topics_list = list(topics.values())
    with open(topics_path, 'wb') as f:
        pickle.dump(topics_list, f)
    
    logger.info('saved topics list to {}'.format(topics_path))

if __name__ == '__main__':
    main()