import nltk
import matplotlib.pyplot as plt
from pprint import pprint
from functools import reduce
from operator import iconcat
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import CRFTagger
from sklearn import metrics


# Maps a file to friendly list like shape:
# [
#  [(word_11:tag_11), (word_12:tag_12), ..., (word_1k1:tag_1k1)],
#  [(word_21:tag_21), (word_22:tag_22), ..., (word_2k2:tag_2k2)],
#  ...
#  [(word_n1:tag_n1), (word_n2:tag_n2), ..., (word_nkn:tag_nkn)]
#]
def process_file(fname:str)->list:
    with open(fname, "r") as f:
        data = f.read()
    data = data.split("\n\n")
    data = list(map(lambda x:x.split("\n"), data))
    data = list(map(lambda x:[tuple(s.split("\t"))[::-1] for s in x], data))
    return data

# Flats a list of list
def to_list(data:list)->list:
    return reduce(iconcat, data, [])

# Splits the words and tags into two lists
def split_words_n_tags(data:list)->tuple:
    words, tags = map(list, zip(*data))
    return words, tags

# Gets just the words from a tagged sentence
def retrive_sents(data:list)->list:
    return list(map(lambda x:[w for w,t in x], data))


train = process_file("engtrain.bio")
test = process_file("engtest.bio")

# Last sentence is empty
train.pop()
test.pop()

# Trains HMM
hmm_tagger = HiddenMarkovModelTrainer().train_supervised(train)

# Trains CRF
crf_tagger = CRFTagger()
crf_tagger.train(train, "model.cfr.tagger")

# Extracts sentences and labels from test
_, labels = split_words_n_tags(to_list(test))
unlabeled_sents = retrive_sents(test)

# Makes predictions using the HMM and the CRF models
hmm_preds = hmm_tagger.tag_sents(unlabeled_sents)
crf_preds = crf_tagger.tag_sents(unlabeled_sents)

# Extracts predictions
_, hmm_preds = split_words_n_tags(to_list(hmm_preds))
_, crf_preds = split_words_n_tags(to_list(crf_preds))

# Classification reports
pprint(metrics.classification_report(labels, hmm_preds))
pprint(metrics.classification_report(labels, crf_preds))


