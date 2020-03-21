import pickle
import argparse
import logging
import spacy
from tqdm import tqdm 
import en_core_web_sm
import os
import json
import re


nlp = en_core_web_sm.load()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

error_mentions = 0


re_token = re.compile(r'[\n\r\t]')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='The tweets pairs to be parsed to XML files')
    parser.add_argument('--text_file', type=str, required=True, help='Output txt file')
    parser.add_argument('--mentions_dir', type=str, required=True, help='mentions json dir')
    #  parser.add_argument('--topics_file', type=str, required=True, help='topics pickled file')
    args = parser.parse_args()
    for arg, value in args._get_kwargs():
        logger.info('{}:\t{}'.format(arg, value))
    parse_to_txt(args.input_file, args.text_file, args.mentions_dir)


def parse_to_txt(input_file, text_file, mentions_dir, rule=None):
    if not os.path.exists(mentions_dir):
        os.makedirs(mentions_dir)
    
    event_mentions_dir = os.path.join(mentions_dir, 'event_mentions')
    entity_mentions_dir = os.path.join(mentions_dir, 'entity_mentions')
    if not os.path.exists(event_mentions_dir):
        os.makedirs(event_mentions_dir)
    
    if not os.path.exists(entity_mentions_dir):
        os.makedirs(entity_mentions_dir)
    
    
    data = pickle.load(open(input_file, 'rb'))
    text = []
    event_mentions = 0
    entity_mentions = 0
    print(rule)
    if rule:
        for rule_data in data:
            rule_name = os.path.basename(rule_data['path']).replace('.json', '')
            if rule_name == rule:
               data = [rule_data]
               break
    logger.info('start running on {} rules'.format(len(data)))
    
    for rule_i, rule_data in enumerate(data):
        rule_name = os.path.basename(rule_data['path']).replace('.pk', '')
        logger.info('start extracting text from {}, {}/{}'.format(rule_name, rule_i+1, len(data)))
        rule_text, rule_entity_mentions, rule_event_mentions = parse_rule_tweets(rule_data['data'], rule_name)
        text.append(rule_text)
        
        event_mentions += len(rule_event_mentions)
        entity_mentions += len(rule_entity_mentions)
        
        save_mentions(os.path.join(event_mentions_dir, rule_name), rule_event_mentions)
        save_mentions(os.path.join(entity_mentions_dir, rule_name), rule_entity_mentions)
        
        logger.info('number of events mentions {}'.format(event_mentions))
        logger.info('number of entities mentions {}'.format(entity_mentions))
        logger.info('finish extracting text from {}'.format(rule_name))
        
    with open(text_file, 'w') as output:
        output.write('\n\n'.join(text))
        output.close()
    logger.info('the text saved successfully to {}'.format(text_file))

    merge_mentions(event_mentions_dir, os.path.join(mentions_dir, 'corpus_Event_gold_mentions.json'))
    merge_mentions(entity_mentions_dir, os.path.join(mentions_dir, 'corpus_Entity_gold_mentions.json'))
    
    
    #  json.dump(entity_mentions, open(os.path.join(mentions_dir, 'corpus_Entity_gold_mentions.json'), 'w'))
    #  json.dump(event_mentions, open(os.path.join(mentions_dir, 'corpus_Event_gold_mentions.json'), 'w'))
    #  logger.info('the mentions saved successfully to {}'.format(mentions_dir))
    logger.info('didn\'t find mentions for {} mentions'.format(error_mentions))


def save_mentions(path, mentions):
    with open(path, 'w') as mf:
        json.dump(mentions, mf)
    del mentions
        

def merge_mentions(mentions_dir, mentions_file):
    files = [os.path.join(mentions_dir, f) for f in os.listdir(mentions_dir)]
    mentions = []
    for mention_file in files:
        with open(mention_file, 'r') as mf:
            mentions += json.load(mf)
    
    with open(mentions_file, 'w') as mf:
        json.dump(mentions, mf)
    

def parse_rule_tweets(rule_data, rule_name):
    text = []
    pair_i = 0
    entity_mentions = []
    event_mentions = []
    rule_name = rule_name.replace('_', '+')

    for tweet_pair in tqdm(rule_data):
        arg0 = tweet_pair['arg0']
        arg1 = tweet_pair['arg1']
        events = tweet_pair['events']
        t1, t2 = tweet_pair['tweets']
        #  create arg:coref_chain dict
        
        #  26/06/2019 - after looking at clustering results, I found out thaa if in arg0 and arg1 there is the same mentio 
        #  (e.g. man (kill\shoot, '979100837917073408', '979115303392051201')) the two mentions recieved the arg1 coref chain
        #  to handle this bug, I added to the key the tweet it related to (0 or 1)
        #  more changes affected by this bug- in parse_tweet function I added a tweet_i parameter (0\1) and changed the keys to the part when the mentions are builded 
        
        '''
        coref_chain = {a:'{}_{}_0'.format(rule_name, pair_i) for a in arg0}
        coref_chain.update({a:'{}_{}_1'.format(rule_name, pair_i) for a in arg1})
        coref_chain.update({e:'{}_{}_event'.format(rule_name, pair_i) for e in events}) 
        '''
        
        coref_chain = {a+str(i):'{}_{}_0'.format(rule_name, pair_i) for i, a in enumerate(arg0)}
        coref_chain.update({a+str(i):'{}_{}_1'.format(rule_name, pair_i) for i, a in enumerate(arg1)})
        coref_chain.update({e:'{}_{}_event'.format(rule_name, pair_i) for e in events}) 
        
        try:
            doc_id1 = '{}_{}{}'.format(t1['id'], pair_i, rule_name)
            doc_id2 = '{}_{}{}'.format(t2['id'], pair_i, rule_name)
            text_t1, mention_t1, ev_mention1 = parse_tweet(t1['tokens'], doc_id1, arg0[0], arg1[0], events[0], coref_chain, 0)
            text_t2, mention_t2, ev_mention2 = parse_tweet(t2['tokens'], doc_id2, arg0[1], arg1[1], events[1], coref_chain, 1)
            text.append(text_t1)
            text.append(text_t2)
            entity_mentions += mention_t1
            entity_mentions += mention_t2
            if ev_mention1:
                event_mentions.append(ev_mention1)
            if ev_mention2:
                event_mentions.append(ev_mention2)
            topic = [doc_id1, doc_id2]
        except Exception as e:
            logger.error('problem at pair {} in rule {} because {}'.format(pair_i, rule_name, str(e)))
        pair_i += 1
        #  topcs.append(topic)
        
    logger.info('extracted text from {} tweet pairs for {}'.format(len(rule_data), rule_name))
    logger.info('extracted {} entity_mentions for {}'.format(len(entity_mentions), rule_name))
    logger.info('extracted {} event_mentions for {}'.format(len(event_mentions), rule_name))
    return '\n\n'.join(text), entity_mentions, event_mentions


def parse_tweet(sent_tokens, doc_id, arg0, arg1, event, arg_coref_chain, tweet_i):
    #  tokenize arg0 and arg1
    global error_mentions
    arg0_t = [str(tok) for tok in nlp(arg0)]
    arg1_t = [str(tok) for tok in nlp(arg1)]
    event_t = [str(tok) for tok in nlp(event)]
    text = []
    entity_mentions = []
    event_mention = None
    tweet_i = str(tweet_i)
    range0 = range1 = range_e = arg0_i = arg1_i = event_i = None
    #  create mentions set parse text in the format: doc_id\tsent_id\ttoken_id\tcoref
    
    
    for sent_i, sent in enumerate(sent_tokens):
        lower_sent = [remove_last_point(t) for t in sent]

        if not arg0_i:
            arg0_i = find_sub_list(arg0_t, lower_sent)
            if arg0_i:
                range0 = range(arg0_i[0], arg0_i[1])
                sent0 = sent_i
                entity_mentions.append({"coref_chain": arg_coref_chain[arg0+tweet_i], 
                                        "doc_id": doc_id, 
                                        "is_continuous": True, 
                                        "is_singleton": False, 
                                        "mention_type": "HUM", 
                                        "score": -1.0, 
                                        "sent_id": sent_i+1, 
                                        "tokens_number": [r for r in range0], 
                                        "tokens_str": ' '.join([t for i, t in enumerate(sent) if i in range0])})
        
        
        if not arg1_i:
            arg1_i = find_sub_list(arg1_t, lower_sent)
            if arg1_i:
                range1 = range(arg1_i[0], arg1_i[1])            
                sent1 = sent_i
                entity_mentions.append({"coref_chain": arg_coref_chain[arg1+tweet_i], 
                                        "doc_id": doc_id, 
                                        "is_continuous": True, 
                                        "is_singleton": False, 
                                        "mention_type": "HUM", 
                                        "score": -1.0, 
                                        "sent_id": sent_i+1, 
                                        "tokens_number": [r for r in range1], 
                                        "tokens_str": ' '.join([t for i, t in enumerate(sent) if i in range1])})
                             
                             
        if not event_i:
            event_i = find_sub_list(event_t, lower_sent)
            if event_i:
                range_e = range(event_i[0], event_i[1])
                sent_e = sent_i
                event_mention = {"coref_chain": arg_coref_chain[event],
                                 "doc_id": doc_id,
                                 "is_continuous": True, 
                                 "is_singleton": False, 
                                 "mention_type": "ACT", 
                                 "score": -1.0,
                                 "sent_id": sent_i+1, 
                                 "tokens_number": [r for r in range_e],
                                 "tokens_str": ' '.join([t for i, t in enumerate(sent) if i in range_e])}
                
                
        for tok_i, token in enumerate(sent):
            coref_chain = '-'
            if arg0_i and tok_i in range0 and sent_i == sent0:
                coref_chain = arg_coref_chain[arg0+tweet_i]
            elif arg1_i and tok_i in range1 and sent_i == sent1:
                coref_chain = arg_coref_chain[arg1+tweet_i]
            elif event_i and tok_i in range_e and sent_i == sent_e:
                coref_chain = arg_coref_chain[event]
            text.append('\t'.join([doc_id, str(sent_i+1), str(tok_i), re_token.sub('', token), coref_chain]))
    #  TODO: Talk to Gabi how to find the tokens number for example
    #  sentence = Terror Hits London, Trump Jr. Rips Mayor
    #  arg0 = london trump jr
    if len(entity_mentions) != 2:
        error_mentions += (2 - len(entity_mentions))
    if not event_mention:
        error_mentions += 1
    return '\n'.join(text), entity_mentions, event_mention


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll
    return None


def remove_last_point(s):
    r = s.lower()
    if r.endswith('.'):
        r = r[::-1].replace('.', '', 1)[::-1]
        return r
    return r
    

if __name__ == '__main__':
    main()
