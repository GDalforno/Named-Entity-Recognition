import os
import operator
import sklearn_crfsuite
from crf_utils import*

def process_conll_file(location:str)->list:
    with open(location, "r") as f:
        data = f.read()
    data = data.split("\n\n")
    data = list(map(lambda x:x.split("\n"), data))
    data.pop()
    data = list(map(lambda x:[operator.itemgetter(*[0, -1])(y.split(" ")) for y in x], data))
    return data

def combine_files(locations:list)->list:
    extended = []
    for f in locations:
        extended.extend(process_conll_file(f))
    return extended

DIR_PL = "../dados-categorias/PLs/"  
DIR_ST = "../dados-categorias/STs/"
DIR_C = "../dados-categorias/Comentarios/"

all_files_pl = [DIR_PL+f for f in os.listdir(DIR_PL)]
all_files_st = [DIR_ST+f for f in os.listdir(DIR_ST)]
all_files_c = [DIR_C+f for f in os.listdir(DIR_C)]

all_files = all_files_pl + all_files_st + all_files_c

all_data = combine_files(all_files)

X = [sent2features(s) for s in all_data]
y = [sent2labels(s) for s in all_data]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.9,
    c2=0.4,
    max_iterations=500,
    all_possible_transitions=True
)
crf.fit(X, y)

joblib.dump(crf, "crf_categories.joblib")