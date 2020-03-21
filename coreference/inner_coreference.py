import shlex, subprocess
import argparse
import logging
import os


COREF_SCRIPT = '../../wd-stanford-coref/wd-stanford-coref/build/libs/stanford-coref-1.0-SNAPSHOT.jar'
COREF_COMMAND = 'java -Xms4096m -Xmx8192m -jar {} -corpus={} -output={}'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=str, required=True, help='The directory of the XML files')
    parser.add_argument('--output_path', type=str, required=True, help='Output coref-chain json files path')
    args = parser.parse_args()
    for arg, value in args._get_kwargs():
        print('{}:\t{}'.format(arg, value))
    inner_coref_directory(args.xml_dir, args.output_path)


def inner_coref_directory(xml_dir, output_path):
    for sub_dir in os.walk(xml_dir):
        path = sub_dir[0]
        if len(path.split('/')) != 3:
            continue
        print(path.split('/'))
        rule = path.split('/')[-1]
        file_name = '{}_wd_coref.json'.format(rule)
        output_file = os.path.join(output_path, file_name)
        logger.info('running wd coref on {} files, save to {}'.format(rule, output_file))
        inner_coref(path, output_file)


def inner_coref(xml_dir, output_file):
    command = COREF_COMMAND.format(COREF_SCRIPT, xml_dir, output_file)
    command = shlex.split(command)
    logger.info('start running {}'.format(command))
    subprocess.run(command)

    
if __name__ == '__main__':
    main()
    