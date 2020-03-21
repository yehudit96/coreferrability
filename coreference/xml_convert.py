import xml.etree.ElementTree as ET
import argparse
import os
import sys
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='The tweets pairs to be parsed to XML files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output XML files directory')
    args = parser.parse_args()
    # Create target Directory if don't exist
    for arg, value in args._get_kwargs():
        print('{}:\t{}'.format(arg, value))
    
    create_xmls(args.input_file, args.output_dir)
    """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
  
    data = pickle.load(open(args.input_file, 'rb'))
    for rule_path, rule_pairs in data:
        rule = os.path.basename(rule_path).replace('.json', '')
        print('start on rule {}'.format(rule))
        path = os.path.join(args.output_dir, rule)
        os.mkdir(path)
        for pair in tqdm(rule_pairs):
            for tweet in pair:
                t_id, text, ents = tweet
                xml_text = parse_tweet_to_xml(nlp(text), t_id)
                file_name = '{}_ecbplus.xml'.format(t_id)
                file_path = os.path.join(path, file_name)
                xml_f = open(file_path, 'w')
                xml_f.write(xml_text)
                xml_f.close()
    """

def create_xmls(input_file, output_dir):
    logger.info('start extracting XMLs for {}'.format(input_file))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    ex_files = os.listdir(output_dir)
    text_data = open(input_file)
    tokens = []
    f_num = 0
    doc_num = 0
    for line in text_data:
        if line == '\n':
            xml_text = parse_tweet_to_xml(tokens, doc_id)
            rule = ''.join([c for c in doc_id if c.isalpha()])
            file_name = '{}_ecbplus.xml'.format(doc_id)
            save_to_file(output_dir, rule, file_name, xml_text)
            f_num += 1
            tokens = []
            doc_num += 1
            continue
        doc_id, sent, tok_id, token, _ = line.split('\t')
        tokens.append((token, sent, tok_id))    
    #  write last doc    
    logger.info(doc_id)
    logger.info(tokens)
    """    
    xml_text = parse_tweet_to_xml(tokens, doc_id)
    file_name = '{}_ecbplus.xml'.format(doc_id)
    if file_name not in ex_files:
        file_path = os.path.join(output_dir, file_name)
        xml_f = open(file_path, 'w')
        xml_f.write(xml_text)
        xml_f.close()
        f_num += 1
    """ 
    logger.info('saved {}/{} files'.format(f_num, doc_num))
    """
    data = pickle.load(open(input_file, 'rb'))
    for rule in data:
        rule = os.path.basename(rule['path']).replace('.json', '')
        print('start on rule {}'.format(rule))
        path = os.path.join(args.output_dir, rule)
        os.mkdir(path)
        for pair in tqdm(rule['data']):
            for tweet in pair['tweets']:
                t_id, text, ents = tweet
                xml_text = parse_tweet_to_xml(tweet['tokens'], tweet['id'])
                file_name = '{}_ecbplus.xml'.format(t_id)
                file_path = os.path.join(path, file_name)
                xml_f = open(file_path, 'w')
                xml_f.write(xml_text)
                xml_f.close()
      """

def parse_tweet_to_xml(tokens, doc_id):
    doc = ET.Element('Document')
    doc.set('doc_name', str(doc_id))
    for tok_text, sent_id, tok_id in tokens:
        t = ET.SubElement(doc, 'token')
        t.set('number', str(tok_id))
        t.set('sentence', str(sent_id))
        t.text = str(tok_text) 
        
    """
    for sent_id, sent in enumerate(text):
        tok_id = 0
        for token in sent:
            tok_text = str(token)
            #  ignore URL and sign tokens
            if tok_text in ['#', '@', '\n', '\t'] or is_url(tok_text):
                continue
            #  remove @ from tokens
            if len(tok_text) > 1 and tok_text.startswith('@'):
                tok_text = tok_text.replace('@', '', 1)

            t = ET.SubElement(doc, 'token')
            t.set('number', str(tok_id))
            t.set('sentence', str(sent_id+1))
            t.text = str(tok_text) 
            tok_id += 1
    """
    return ET.tostring(doc).decode()

def save_to_file(dir_path, rule, f_name, xml_text):
    file_path = os.path.join(dir_path, rule)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        logger.info('created {}'.format(file_path))
    
    file_path = os.path.join(file_path, f_name)
    xml_f = open(file_path, 'w')
    xml_f.write(xml_text)
    xml_f.close()
    
    
if __name__ == '__main__':
    main()