import argparse
import logging
import pickle
from newspaper import Article
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from tqdm import tqdm
import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


ID = 'id'
ARTICLE = 'article'
URLS = 'urls'
bad_starts = ['https://twitter.com', 'http://fb.me', 'https://www.facebook.com']
bad_ends = ['.com', '.com/', '.me', '.me/']
max_workers = 80
creation_date = datetime.date(2020, 2, 27)

def main():
    logger.info('starting..')
    extract_directory(args.rules_path)

"""
    tweets_pair = pickle.load(open(args.tweets_path, 'rb'))
    tweets_pair = tweets_pair[:args.limit_examples] if args.limit_examples else tweets_pair
    for pair in tqdm(tweets_pair):
        for tweet in pair['tweets']:
            for url in tweet['urls']:
                if not is_valid_url(url['link']):
                    continue
                text = get_article(url['link'])
                if text:
                    url['text'] = text
    pickle.dump(tweets_pair, open(args.output_path, 'wb'))
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The file path for the tweet pairs file")

    global args
    args = parser.parse_args()
    logger.info("start extracting articles")
    logger.info("tweets file path:\t{}".format(args.rules_path))


def extract_directory(rules_dir):
    rules_files = [os.path.join(rules_dir, f) for f in os.listdir(rules_dir)] # if not_updated(os.path.join(rules_dir, f))]
    for i, rule_file in enumerate(rules_files):
        logger.info('{}/{} parsing urls for {}'.format(i+1, len(rules_files), rule_file))
        extract_rule(rule_file)
        

def extract_rule(rule_path):
    start = time()
    with open(rule_path, 'rb') as rf:
        tweets_pair = pickle.load(rf)
    urls = {}
    for i, pair in tqdm(enumerate(tweets_pair)):
        for j, tweet in enumerate(pair['tweets']):
            for k, url in enumerate(tweet['urls']):
                if not is_valid_url(url['link']):
                    continue
                urls[url['link']] = [i, j, k]
    if len(urls) > 0:
        arts = []
        mw = min(max_workers, len(urls))
        logger.info('max workers:{}'.format(mw))
        logger.info('urls to parse:{}'.format(len(urls)))
        with ThreadPoolExecutor(max_workers=mw) as executor:
            future_to_url = {executor.submit(get_article, url): url for url in urls.keys()}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    text = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    arts.append([urls[url], text])
        end = time()
        for inx, article in arts:
            if article == '':
                continue
            i, j, k = inx
            tweets_pair[i]['tweets'][j]['urls'][k]['text'] = article
        logger.info(f'Time taken: {time() - start}')
    with open(rule_path, 'wb') as rf:
        pickle.dump(tweets_pair, rf)
    logger.info('extracted article successfully for {}'.format(rule_path))
    
    
def get_article(link):
    try:
        article = Article(link)
        article.download()
        article.parse()
    except:
        return ''
    return article.text


def is_valid_url(url, check_redirect=True):
    """
    for s in bad_starts:
        if url.startswith(s):
            if not check_redirect:
                return False
            re_url = get_redirects(url)
            if re_url != url:
                is_valid_url(re_url, False)
            else:
                return False  
    """
    for e in bad_ends:
        if url.endswith(e):
            return False
    if url[:-2].endswith('.co.'):
        return False
    return True


def get_redirects(link):
    response = requests.get(link)
    if len(response.history) <= 2:
        return link
    return response.history[-1].url
    

def not_updated(path):
    modified = datetime.date.fromtimestamp(os.path.getmtime(path))
    return modified == creation_date


if __name__ == '__main__':
    parse_args()
    main()
