import sys
sys.path.append('../')
from NER.NER_similarity import calculate_ner_sim
import argparse
import logging
import json
import pickle
from tqdm import tqdm
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    files = os.listdir(args.data_dir)
    for i, data_file in enumerate(files):        
        logger.info('{}/{} start calculating NE similarity for {}'.format(i+1, len(files), data_file))
        calculate_rule_ne_sim(os.path.join(args.data_dir, data_file))    
    

def calculate_rule_ne_sim(path):
    #  logger.info('start calculating NE similarity for {}'.format(path))
    with open(path, 'rb') as tweets_file:
        tweets = pickle.load(tweets_file)
        articles_num = 0
        for pair in tqdm(tweets):
            tweet1, tweet2 = pair['tweets']
            text1, text2 = get_max_text(tweet1['urls']), get_max_text(tweet2['urls'])
            if 'features' not in pair:
                pair['features'] = dict()
            if text1 and text2:
                pair_text = {'tweet1': tweet1['text'],
                             'tweet2': tweet2['text'],
                             'article1': text1,
                             'article2': text2}
                sim = calculate_ner_sim(pair_text)
                pair['features']['NE'] = sim
                articles_num += 1
                
    logger.info('calculated NE similarity for {}/{}'.format(articles_num, len(tweets)))
    pickle.dump(tweets, open(path, 'wb'))
    logger.info('end calculating NE similarity for {}'.format(path))
        

def get_max_text(urls):
    texts = []
    for url in urls:
        if 'text' in url:
            texts.append(url['text'])
    if len(texts) == 0:
        return None
    return max(texts, key=lambda t:len(t))

def rule_ne_similarity(path):
    
    tweets = json.load(open(path))
    sim_values = {}
    logger.info('{} tweets pairs was extracted from {}'.format(len(tweets), path))
    for i, pair in tqdm(enumerate(tweets)):
        pair_values = []
        pair_texts = []
        tweet1, tweet2 = pair['tweets']
        for url1 in tweet1['urls']:
            for url2 in tweet2['urls']:
                if 'text' in url1 and 'text' in url2:
                    pair_texts.append({'tweet1': tweet1['text'],
                                       'tweet2': tweet2['text'],
                                       'article1': url1['text'],
                                       'article2': url2['text']})
        for text in pair_texts:
            sim = calculate_ner_sim(text)
            pair_values.append(sim)
        sim_values[i] = pair_values
    return sim_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The data pickle files directory")

    global args
    args = parser.parse_args()
    logger.info("start running on articles data to NER similarity")
    logger.info("articles path:\t{}".format(args.data_dir))


if __name__ == '__main__':
    parse_args()
    main()