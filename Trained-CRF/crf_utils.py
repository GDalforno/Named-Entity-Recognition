import joblib
import nltk
from nltk.corpus import PlaintextCorpusReader 
from nltk import sent_tokenize, word_tokenize, pos_tag 
from nltk.stem import WordNetLemmatizer


#função do tamanho da palavra (returna True se for maior que 4)
def length(word):
    if len(word) >= 4: 
        tamanho = True
    else:
        tamanho = False
    return tamanho

teste_tagger = joblib.load('POS_tagger_brill.pkl')
def postag(word):
    phrase = word
    postag = teste_tagger.tag(word_tokenize(phrase))
    return postag[0][1]

#tamanho da setenca
def tamsent(sent,i):
    conta = []
    valor = []
    for i in range(len(sent)):
        conta.append(sent[i].count(sent[i][0]))
    valor = sum(conta)
    return valor

#frequencia da palavra na sentenca
def freqwordsent(sent,word):
    conta = []
    valor = []
    for j in range(len(sent)):
        conta.append(sent[j].count(word))
    valor = sum(conta)
    return valor


def word2features(sent, i):
    word = sent[i][0]  
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-1:]': word[-1:],        
        'word[:1]': word[:1],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag(word),
        'postag[:2]': postag(word)[:1],
        'postag[:2]': postag(word)[:2],
        'tamanho': length(word),
        'word.isalnum()' : word.isalnum(),
        'len(word)': len(word),
        'tamanho(sent)': tamsent(sent,i),
        'freqwordsent' : freqwordsent(sent,word),   
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag(word1),
            '-1:postag[:2]': postag(word1)[:1],
            '-1:postag[:2]': postag(word1)[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word[-1:]': word1[-1:],
            '-1:word[:1]': word1[:1],
            '-1:word[:2]': word1[:2],
            '-1:word[:3]': word1[:3],
            '-1:len(word)': len(word1),
            '-1:word.isalnum()' : word1.isalnum(),
        })
    else:
        features['Inicio'] = True

    if i > 1:
        word2 = sent[i-2][0]
        features.update({
            '-2:word': word2,
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag(word2),
            '-2:postag[:2]': postag(word2)[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word[-1:]': word2[-1:],
            '-2:word[:1]': word2[:1],
            '-2:word[:2]': word2[:2],
            '-2:word[:3]': word2[:3],
            '-2:len(word)': len(word2),
            '-2:word.isalnum()' : word2.isalnum(),

        })
    if i < len(sent)-1:
        word3 = sent[i+1][0]
        features.update({
            '+1:word': word3,
            '+1:word.lower()': word3.lower(),
            '+1:word.istitle()': word3.istitle(),
            '+1:word.isdigit()': word3.isdigit(),
            '+1:word.isupper()': word3.isupper(),
            '+1:postag': postag(word3),
            '+1:postag[:2]': postag(word3)[:2],
            '+1:word[-3:]': word3[-3:],
            '+1:word[-2:]': word3[-2:],
            '+1:word[-1:]': word3[-1:],
            '+1:word[:1]': word3[:1],
            '+1:word[:2]': word3[:2],
            '+1:word[:3]': word3[:3],
            '+1:len(word)': len(word3),
            '+1:word.isalnum()' : word3.isalnum()
            })
    else:
        features['Final'] = True
   
    if i < len(sent)-2:
        word4 = sent[i+2][0]
        features.update({
            '+2:word': word4,
            '+2:word.lower()': word4.lower(),
            '+2:word.istitle()': word4.istitle(),
            '+2:word.isdigit()': word4.isdigit(),
            '+2:word.isupper()': word4.isupper(),
            '+2:postag': postag(word4),
            '+2:postag[:2]': postag(word4)[:2],
            '+2:word[-3:]': word4[-3:],
            '+2:word[-2:]': word4[-2:],
            '+2:word[-1:]': word4[-1:],
            '+2:word[:1]': word4[:1],
            '+2:word[:2]': word4[:2],
            '+2:word[:3]': word4[:3],
            '+2:len(word)': len(word4),
            '+2:word.isalnum()' : word4.isalnum()
        })     

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

lista = [
        'bias',
        'word',
        'word.lower()',
        'word[-3:]',
        'word[-2:]',
        'word[-1:]',
        'word[:1]',
        'word[:2]',
        'word[:3]',
        'word.isupper()',
        'word.istitle()',
        'word.isdigit()',
        'postag','postag[:2]',
        'postag[:2]',
        'tamanho',
        'word.isalnum()',
        'len(word)',
        'tamanho(sent)',
        'freqwordsent',
        '-1:word',
        '-1:word.lower()',
        '-1:word.istitle()',
        '-1:word.isdigit()',
        '-1:word.isupper()',
        '-1:postag',
        '-1:postag[:2]',
        '-1:postag[:2]',
        '-1:word[-3:]',
        '-1:word[-2:]',
        '-1:word[-1:]',
        '-1:word[:1]',
        '-1:word[:2]',
        '-1:word[:3]',
        '-1:len(word)',
        '-1:word.isalnum()',
        '-2:word',
        '-2:word.lower()',
        '-2:word.istitle()',
        '-2:word.isdigit()',
        '-2:word.isupper()',
        '-2:postag',
        '-2:postag[:2]',
        '-2:word[-3:]',
        '-2:word[-2:]',
        '-2:word[-1:]',
        '-2:word[:1]',
        '-2:word[:2]',
        '-2:word[:3]',
        '-2:len(word)',
        '-2:word.isalnum()',
        '+1:word',
        '+1:word.lower()',
        '+1:word.istitle()',
        '+1:word.isdigit()',
        '+1:word.isupper()',
        '+1:postag',
        '+1:postag[:2]',
        '+1:word[-3:]',
        '+1:word[-2:]',
        '+1:word[-1:]',
        '+1:word[:1]',
        '+1:word[:2]',
        '+1:word[:3]',
        '+1:len(word)',
        '+1:word.isalnum()',
        '+2:word',
        '+2:word.lower()',
        '+2:word.istitle()',
        '+2:word.isdigit()',
        '+2:word.isupper()',
        '+2:postag',
        '+2:postag[:2]',
        '+2:word[-3:]',
        '+2:word[-2:]',
        '+2:word[-1:]',
        '+2:word[:1]',
        '+2:word[:2]',
        '+2:word[:3]',
        '+2:len(word)',
        '+2:word.isalnum()'
]

