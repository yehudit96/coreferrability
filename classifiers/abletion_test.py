import json
import pickle
import numpy as np

from random_forest_classifier import random_forest, score_list, calculate_score
from random_forest_params import best_params_randomize_search


features_groups = {'NE': ["NE_0.26", "NE_0.26_avg", "NE_0.26_event_pc"],
   'chirps': ["chirps_days", "chirps_max_rank", "chirps_num", "chirps_rules_num", "day_num", "pairs_num"],
   'component': ["component_avg", "component_num", "day_num"],
   'coref': ["entity_ipc", "entity_pc", "entity_wc", "event_ipc", "event_pc"],
   'clique': ["in_clique"],
   #'chirps_org': ["day_num", "pairs_num"],
   'all': []}

data_path = 'datasets/Kian/{}_an'

features_path = 'features.json'

with open(features_path, 'r') as f_in:
    features = json.load(f_in)
features = {f: i for i, f in enumerate(features)}

datasets = {}
splits = ['train', 'dev', 'test']
for split in splits:
    with open(data_path.format(split), 'rb') as f:
        datasets[split] = pickle.load(f)
    print(f'{split} size {len(datasets[split])}')
    
    
def get_features_index(features_name):
    return [features[feature] for feature in features_name]


def remove_features(dataset, indexes):
    rules, x, y = zip(*dataset)
    new_X = np.array(x)
    new_X = np.delete(x, indexes, axis=1)
    if new_X.shape == np.array(x).shape:
        print('error at remove_features')
    return list(zip(rules, new_X, y))


def abletion_test():
    for feature_group, features_name in features_groups.items():
        indexes = get_features_index(features_name)
        abletion_sets = {}
        for split in splits:
            abletion_sets[split] = remove_features(datasets[split], indexes)
        print(f'without {feature_group}')
        clf = random_forest(abletion_sets['train'], abletion_sets['dev'])
        print('----------')
        

if __name__ == '__main__':
    abletion_test()