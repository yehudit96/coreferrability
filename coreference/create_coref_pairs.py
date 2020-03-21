import spacy
import os
import json
import pickle
import re
from tqdm import tqdm
import argparse
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


nlp = spacy.load('en_core_web_sm')

#  PATHS = ['../data/tweets/good_tweets', '../data/tweets/bad_tweets']
OUT_PATH = 'coref_corpus_args.pk'
"""
This scrip create a one-file data for all the tweets we collected (appears in the args.good_dir and args.bad_dir)
The data is is the format:
A list of data per each rule, each item in the list is a dictionary
{
  path:the path the data extracted from
  data:list of dictionries for each tweets pair
      tweets: list of dictionaries for each tweet
          id:tweet id
          text:the raw tweet's text
          tokens:tokenized clean text splited to sentences (extract_sents)
          created_at:the date the tweet was created
      arg0:the arg0s of the tweets (the 1st refer to the 1st tweet, the 2nd to the 2nd tweet)
      arg1:as arg0 
      event:the events (paredicate paraphrases) of the tweets
}
"""

def main():
    parser = argparse.ArgumentParser()
    #  parser.add_argument('--good_dir', type=str, required=True, help='Directory of good pairs data')
    parser.add_argument('--bad_dir', type=str, required=True, help='Directory of bad pairs data')
    parser.add_argument('--out_file', type=str, required=True, help='output tweets file path')
    args = parser.parse_args()
    for arg, value in args._get_kwargs():
        logger.info('{}:\t{}'.format(arg, value))
    
#    PATHS = [args.good_dir, args.bad_dir]
    PATHS = [args.bad_dir]
    files = []
    for path in PATHS:
        files += [os.path.join(path, f) for f in os.listdir(path)]
    data = []
    for i, f in enumerate(files):
        logger.info('start extracting from {} {}/{}'.format(f, i, len(files)))
        data.append({'path':f, 'data':extract_tweets_data(f)})
    
    pickle.dump(data, open(args.out_file, 'wb'))


def extract_directory_data(paths, out_file):
    files = []
    for path in paths:
        files += [os.path.join(path, f) for f in os.listdir(path)]
    data = []
    for i, f in enumerate(files):
        logger.info('start extracting from {} {}/{}'.format(f, i, len(files)))
        data.append({'path':f, 'data':extract_tweets_data(f)})
    
    pickle.dump(data, open(out_file, 'wb'))
        

def extract_tweets_data(path):
    with open(path, 'rb') as f:
        tweets = pickle.load(f)
    tweets_e = []
    for pair in tqdm(tweets):
        tweets_p = {}
        tweets_p['tweets'] = []
        for tweet in pair['tweets']:
            t_id = tweet['id']
            text = tweet['text']
            sent = extract_sents(tweet['text'])
            date = tweet['created_at']
            tweets_p['tweets'].append({'id':t_id, 'text':text, 'tokens':sent, 'created_at':date})
        t1, t2 = pair['tweets']
        arg0 = [t1['arg0'], t2['arg0']]
        arg1 = [t1['arg1'], t2['arg1']]
        tweets_p['arg0'] = arg0
        tweets_p['arg1'] = arg1
        events = [t1['origin_rule'], t2['origin_rule']]
        tweets_p['events'] = [clean_event(e) for e in events]
        tweets_e.append(tweets_p)
    return tweets_e



def extract_sents(tweet):
    doc = nlp(tweet)
    sents = []
    for sent in doc.sents:
        t_list = []
        for token in sent:
            tok_text = str(token)
            if tok_text in ['#', '@'] or is_url(tok_text):
                continue
            if len(tok_text) > 1 and tok_text.startswith('@'):
                tok_text = tok_text.replace('@', '', 1)
            t_list.append(tok_text)
        if t_list:
            sents.append(t_list)
    return sents


def is_url(token):
    """
    check if the token is a url
    :param token: the token to check on
    :return: if the token is a URL: True, if not: False
    """
    if token in ['https//', 'http//']:
        return True
        
    url = re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] | [! * \(\),] | (?: %[0-9a-fA-F][0-9a-fA-F]))+',
                    token)
    return bool(url)
    

def clean_event(event):
    new_event = event.replace('{a0}', '').replace('{a1}', '')
    return ' '.join(new_event.split())



if __name__ == '__main__':
    main()
