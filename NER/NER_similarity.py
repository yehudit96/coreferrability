import en_core_web_sm
from fuzzywuzzy import fuzz
from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib.pylab as plt
import xlrd
from collections import OrderedDict
import numpy as np

import argparse
import logging
import pickle

nlp = en_core_web_sm.load()  # en_vectors_web_lg.load()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'EVENT', 'PRODUCT']
RATIO_TS = 90


def calculate_ner_sim(pair):
    '''
    recieves pairs in format of dictionary:
    { 
      'tweet1': tweet1['text'],
      'tweet2': tweet2['text'],
      'article1': url1['text'],
      'article2': url2['text']
    }
    and return the similarity between the two pairs
    '''
    sim, clusters = match_pair(pair, max_similarity)
    return sim


def max_similarity(clusters_a, clusters_b):
    """
    calclate the cluster with the max covarege with the other cluster
    """
    if not len(clusters_a) or not len(clusters_b):
        return 0.0
    i_a = [False] * len(clusters_a)
    i_b = [False] * len(clusters_b)
    for i, a in enumerate(clusters_a):
        for j, b in enumerate(clusters_b):
            if can_merge(a , b) > 0:
                i_a[i] = True
                i_b[j] = True
    return max([float(sum(i_a))/len(i_a), float(sum(i_b)/len(i_b))])


def get_texts(pair):
    """
    extract the needed text from the articles
    """
    concat = '{}\n{}'
    par1 = extract_n_paragraph(pair['article1'], "\n", 3)
    par2 = extract_n_paragraph(pair['article2'], "\n", 3)
    text1 = concat.format(pair['tweet1'], par1)
    text2 = concat.format(pair['tweet2'], par2)

    return text1, text2


def extract_n_paragraph(text, det, n):
    pars = text.split(det)
    pars = [p for p in pars if p] #remove empty strings
    pars = pars[:n]
    return det.join(pars)


def extract_ents(ner):
    """
    extract and clear only relevant entities
    """
    return [X.text[:-2] if X.text.endswith("'s") or X.text.endswith("’s") else X.text for X in ner.ents if
            X.text not in ["'s", "’s"] and X.label_ in LABELS]


def cluster_doc(ents):
    """
    cluster together corefer entities
    """
    clusters = [[e] for e in set(ents)]
    merge = True
    while merge:
        merge = False
        l = len(clusters)
        for i in range(l):
            for j in range(i + 1, l):
                if can_merge(clusters[i], clusters[j]) == 1:
                    [clusters[i].append(e) for e in clusters[j]]
                    clusters[j].clear()
                    merge = True
        clusters = [c for c in clusters if c]

    return clusters


def can_merge(c1, c2):
    """
    gets two cluster and check weather thy can be merged
    return the number of matched pair out of possible matches
    """
    if not c1 or not c2:
        return False
    good = total = 0.0
    for e in c1:
        for f in c2:
            # TODO: add acronym handling
            if fuzz.WRatio(e, f) >= RATIO_TS:
                good += 1
            total += 1
    return good / total


def match_pair(pair, sim_method):
    """
    calculate the similarity value for a given pair using similarity methos
    """
    doc1, doc2 = get_texts(pair)
    ents1 = extract_ents(nlp(doc1))
    ents2 = extract_ents(nlp(doc2))
    #  cluster the corefer entities for each document
    c1 = cluster_doc(ents1)
    c2 = cluster_doc(ents2)
    similarity = sim_method(c1, c2)
    return similarity, [c1, c2]
