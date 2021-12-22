import joblib
from nltk.tokenize import word_tokenize
from crf_utils import sent2features

# Carrega o modelo treinado
crf = joblib.load("crf_types.joblib")  # Modelo para tagear PL's, ST's e Comentarios com tipos
# crf = joblib.load("crf_categories.joblib")  # Modelo para tagear PL's, ST's e Comentarios com categorias 

# Exemplo de input
sentence = "Apresentação do Projeto de Lei n. 4593/2021, pela Deputada Tabata Amaral (PSB), que 'Acrescenta parágrafo ao art. 37 da Lei nº 9.394, de 1996, de diretrizes e bases da educação nacional, para assegurar às mulheres com filhos ou dependentes a oferta de vagas, no turno diurno, para cursarem a educação de jovens e adultos. "

# Transforma a sentenca em uma lista de tokens
tokenized_sentence = [(x,) for x in word_tokenize(sentence)]

# Transforma a lista de tokens num elemento do feature space do CRF
X = [sent2features(tokenized_sentence)]

# Faz a predição das tags
y = crf.predict(X)

# Faz o tageamento da sentença
tagged_sentence = [(x[0], y) for x, y in zip(tokenized_sentence, y[0])]

# Imprimi o resultado
for word, tag in tagged_sentence:
    print(f"({word}, {tag})")
