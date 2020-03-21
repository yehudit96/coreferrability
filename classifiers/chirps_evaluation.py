import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main():
    with open('datasets/Kian/eval_an', 'rb') as f:
        eval_rules = pickle.load(f)

    rules, x, y = zip(*eval_rules)
    with open('best_rf', 'rb') as f_clf:
        clf = pickle.load(f_clf)

    pred = clf.predict(x)
    pred_proba = clf.predict_proba(x)
    proba = list(zip(x,y,rules, pred_proba, pred))
    sort_proba = sorted(proba, key=lambda a:a[3][1], reverse=True)
    _, y, _, _, pred = zip(*sort_proba)
    inx = list(map(int, np.linspace(0,len(y), 5)))
    bins = list(zip(inx[:-1], inx[1:])) 
    print('sort by classifies')
    for bin_s, bin_e in bins:
        print("number of positive rules between {} to {}".format(bin_s, bin_e))
        print(sum(y[bin_s:bin_e]))
    
    sort_chirps = sorted(proba, key=lambda a:a[0][4], reverse=True)
    _, y, _, _, pred = zip(*sort_chirps)
    print('sort by chirps rank')
    for bin_s, bin_e in bins:
        print("number of positive rules between {} to {}".format(bin_s, bin_e))
        print(sum(y[bin_s:bin_e]))
    

if __name__ == "__main__":
    main()