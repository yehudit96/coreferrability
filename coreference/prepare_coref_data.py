import sys
sys.path.append('../../wd-plus-srl-extraction/src')
sys.path.append('../../wd-plus-srl-extraction/')

import create_coref_pairs
import tweet_text
import srl_allen
import xml_convert
import inner_coreference
import topics_list

import os
import argparse
import json  
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # config file can be found in prepare_config dir
    parser.add_argument('--config', type=str, required=True, help='prepare coref data configuration (json file)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    check_config(config)
    
    # stage 1
    if config['starting_stage'] == 1:
        logger.info('start creating coref pairs')
        data_dirs = filter(lambda d: os.path.isdir(d), map(lambda f: os.path.join(config['data_dir'], f), os.listdir(config['data_dir'])))
        create_coref_pairs.extract_directory_data(data_dirs, config['corpus_path'])
        
        logger.info('created coref pairs file')
    
    # stage 2
    if config['starting_stage'] <= 2:
        logger.info('creating mentions and text files')
        tweet_text.parse_to_txt(config['corpus_path'], config['text_path'], config['mentions_dir'])
        logger.info('created mentions and text files')
    
    # stage 3
    if config['starting_stage'] <= 3:
        logger.info('creating topics list file')
        topics_list.create_topics_list(config['text_path'], config['topics_file'])
        logger.info('created topics list file')
    
    # stage 4
    if config['starting_stage'] <= 4:
        logger.info('creating srl file')
        srl_allen.create_srl_tweets(config['corpus_path'], config['srl_path'])
        logger.info('created srl file')  
    
    # stage 5
    if config['starting_stage'] <= 5:
        logger.info('creating xml files')
        xml_convert.create_xmls(config['text_path'], config['xmls_dir'])
        logger.info('created xml files')

    # stage 6
    if config['starting_stage'] <= 6:
        logger.info('running inner coerference')
        inner_coreference.inner_coref_directory(config['xmls_dir'], config['inner_coref_dir'])
        logger.info('created inner coreference')
    
    # stage 7
    if config['starting_stage'] <=7:
        logger.info('marging inner coref files')
        inner_files = map(lambda x: os.path.join(config['inner_coref_dir'], x), os.listdir(config['inner_coref_dir']))
        all_coref = []
        for f in inner_files:
            with open(f) as inner_f:
                coref = json.load(inner_f)
            all_coref += coref
        
        with open(config['inner_coref_file'], 'w') as inner_a:
            json.dump(all_coref, inner_a)
        logger.info('marged inner coref files')
  

def check_config(config):
    keys = ['data_dir', 'corpus_path', 'text_path', 'mentions_dir', 'topics_file', 'srl_path', 'xmls_dir', 'inner_coref_dir', 'inner_coref_file', 'starting_stage']
    for key in keys:
        if key not in config:
            raise Exception('{} not in config file'.format(key))
        else:
            logger.info('{}:\t{}'.format(key, config[key]))


if __name__ == '__main__':
    main()
