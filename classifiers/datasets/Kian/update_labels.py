import pickle
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, type=str, required=True, help="split to update")
    args = parser.parse_args()

    rules_annotations = {}
    for data_split in ['train', 'dev', 'test', 'eval']:
        with open('../MTAnnotation/results_{}/final_results'.format(data_split), 'rb') as f:
            rules_annotations.update(pickle.load(f))
    with open('datasets/Kian/{}'.format(args.split), 'rb') as f_in:
        data = pickle.load(f_in)
    new_data = []
    n = 0
    for pair in data:
        rule = tuple(sorted(pair[0].split('_')))
        # if the pair labaled as positive
        if pair[2] == 1:
            new_data.append(pair)
        # if the pair labaled as negative        
        elif pair[2] == 0 and rule in rules_annotations:
            pair = list(pair)
            pair[2] = int(rules_annotations[rule])
            pair = tuple(pair)
            new_data.append(pair)
        else:
            n+= 1
    print('new dataset length:{}'.format(len(new_data)))
    print('positive:{}, negative:{}'.format(sum(map(lambda x:x[2], new_data)), len(new_data) - sum(map(lambda x:x[2], data))))
    print('not found: {}'.format(n))
    with open('datasets/Kian/{}_an'.format(args.split), 'wb') as f_out:
        pickle.dump(new_data, f_out)
    

if __name__ == "__main__":
    main()