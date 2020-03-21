import pandas
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import argparse

def main():
    """
    cmpute the label for each rule (pos\neg) by the MT annotations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file",
                        default=None,
                        type=str,
                        required=True,
                        help="all rules results file")
    parser.add_argument("--out",
                        default=None,
                        type=str,
                        required=True,
                        help="path of out computed results pickeld file")
    
    args = parser.parse_args()
    results = load_results(args.results_file)
    labels = label_rules(results)
    print('number of rules: {}'.format(len(labels)))
    print(np.bincount(list(labels.values())))

    with open(args.out, 'wb') as f_out:
        pickle.dump(labels, f_out)

def load_results(result_file):
    """
    Load the batch results from the CSV
    :param result_file: the batch results CSV file from MTurk
    :return: answers, comments and real label (is exists) for each rule
    """
    rules_answers = defaultdict(dict)
    table = pandas.read_csv(result_file, dtype={'Input.tweet_id2': str})
    print('table len: {}'.format(table.shape))
    i = 0 
    for index, row in table.iterrows():

        hit_id = row['HITId']
        worker_id = row['WorkerId']

        # Input fields
        p1 = row['Input.p1']
        p2 = row['Input.p2']

        # Answer fields
        answer = any([row['Answer.ans%d' % (i + 1)] == 'yes' for i in range(3)])
        answers = [True if row['Answer.ans%d' % (i + 1)] == 'yes' else False for i in range(3)]
        comment = row['Answer.comment']

        # Annotation field
        annotation = row['Input.annotated'].lower()
        
        key = tuple(sorted((p1, p2)))

        if 'answers' not in rules_answers[key]:
            rules_answers[key]['answers'] = {}
        rules_answers[key]['answers'][worker_id] = answers
        if comment:
            if 'comments' not in rules_answers[key]:
                rules_answers[key]['comments'] = {}
            rules_answers[key]['comments'][worker_id] = comment

        if annotation == 'none':
            rules_answers[key]['annotation'] = None
        else:
            rules_answers[key]['annotation'] = True if annotation == 'true' else False

        if (annotation == 'true' and answer == False) or (annotation == 'false' and answer == True):
            #  print('worker {} annotate {} wrongly, {} instead {}'.format(worker_id, key, answer, annotation.capitalize()))
            i += 1
    print(i)
    return rules_answers


def label_rules(results):
    rules_label = {}
    gold = []
    predicted = []
    for rule, result in results.items():
        annotation = result['answers'].values()
        instances_answers = zip(*annotation)
        answers = [compute_majority_gold(ans) for ans in instances_answers]
        rules_label[rule] = any(answers)
        #rules_label[rule] = compute_majority_gold([any(worker_ans) for worker_ans in annotation])
        if result['annotation'] != None:
            predicted.append(rules_label[rule])
            gold.append(result['annotation'])
            if rules_label[rule] != result['annotation']:
                print('{}\tg:{}\tp:{}'.format(rule, result['annotation'], rules_label[rule]))
    print(precision_recall_fscore_support(gold, predicted))
    return rules_label
        

def compute_majority_gold(answers):
    return bool(np.argmax(np.bincount(answers)))

if __name__ == '__main__':
    main()