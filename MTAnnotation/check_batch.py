import math
import pickle
import pandas
import numpy as np
import argparse
from collections import defaultdict, Counter
from sklearn.metrics import cohen_kappa_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file",
                        type=str,
                        default=None,
                        required=True,
                        help="The batch results csv file")

    args = parser.parse_args()

    workers, results, workers_count = load_results(args.results_file)
    kappa, workers_to_remove = cohens_kappa(results, workers)
    logging.info('cohens kappa score: {}'.format(kappa))
    # if len(workers_to_remove) > 0:
    logging.info('{} workers need to be removed'.format(len(workers_to_remove)))
    logging.info('\n'.join(['{}: {}'.format(w, workers_count[w])for w in workers_to_remove]))


def load_results(result_file):
    """
    Load the batch results from the CSV
    :param result_file: the batch results CSV file from MTurk
    :return: the workers and the answers
    """
    worker_answers = {}
    workers = set()
    workers_counter = Counter()
    table = pandas.read_csv(result_file, dtype={'Input.tweet_id2': str})

    for index, row in table.iterrows():

        hit_id = row['HITId']
        worker_id = row['WorkerId']

        # Input fields
        p1 = row['Input.p1']
        p2 = row['Input.p2']

        # Answer fields
        answer = any([row['Answer.ans%d' % (i + 1)] == 'yes' for i in range(3)])
        comment = row['Answer.comment']

        key = (p1, p2)

        if key not in list(worker_answers.keys()):
            worker_answers[key] = {}

        annotation = row['Input.annotated']
        if (annotation == 'True' and answer == False) or (annotation == 'False' and answer == True):
            logger.info('worker {} annotate {} wrongly, {} instead {}'.format(worker_id, key, answer, annotation))
            for i in range(1, 4):
                print('{}\n{}'.format(row['Input.inst{}_1'.format(i)],
                                      row['Input.inst{}_2'.format(i)]))
        workers.add(worker_id)
        workers_counter.update([worker_id])
        worker_answers[key][worker_id] = (answer, comment)

    return workers, worker_answers, workers_counter


def cohens_kappa(results, workers):
    """
    Compute Cohen's Kappa on all workers that answered at least 5 HITs
    :param results:
    :return:
    """
    answers_per_worker = {worker_id: {key: results[key][worker_id] for key in list(results.keys())
                                      if worker_id in results[key]}
                          for worker_id in workers}
    answers_per_worker = {worker_id: answers for worker_id, answers in answers_per_worker.items()
                          if len(answers) >= 5}
    curr_workers = list(answers_per_worker.keys())
    worker_pairs = [(worker1, worker2) for worker1 in curr_workers for worker2 in curr_workers if worker1 != worker2]

    label_index = {True: 1, False: 0}
    pairwise_kappa = {worker_id: {} for worker_id in list(answers_per_worker.keys())}

    # Compute pairwise Kappa
    for (worker1, worker2) in worker_pairs:

        mutual_hits = set(answers_per_worker[worker1].keys()).intersection(set(answers_per_worker[worker2].keys()))
        mutual_hits = set([hit for hit in mutual_hits if not pandas.isnull(hit)])
        mutual_hits = list(mutual_hits)
        if len(mutual_hits) >= 5:

            worker1_labels = np.array([label_index[answers_per_worker[worker1][key][0]] for key in mutual_hits])
            worker2_labels = np.array([label_index[answers_per_worker[worker2][key][0]] for key in mutual_hits])
            curr_kappa = cohen_kappa_score(worker1_labels, worker2_labels)
            if not math.isnan(curr_kappa):
                pairwise_kappa[worker1][worker2] = curr_kappa
                pairwise_kappa[worker2][worker1] = curr_kappa

    # Remove worker answers with low agreement to others
    workers_to_remove = set()

    for worker, kappas in pairwise_kappa.items():
        if np.mean(list(kappas.values())) < 0.1:
            print('Removing %s' % worker)
            workers_to_remove.add(worker)

    kappa = np.mean([k for worker1 in list(pairwise_kappa.keys()) for worker2, k in pairwise_kappa[worker1].items()
                     if not worker1 in workers_to_remove and not worker2 in workers_to_remove])

    # Return the average
    return kappa, workers_to_remove


if __name__ == '__main__':
    main()
