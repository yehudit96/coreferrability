import pickle

PATH = 'datasets/Kian/{}_an'

with open(PATH.format('train'), 'rb') as f:
    train = pickle.load(f)
    
with open(PATH.format('dev'), 'rb') as f:
    dev = pickle.load(f)

with open(PATH.format('test'), 'rb') as f:
    test = pickle.load(f)