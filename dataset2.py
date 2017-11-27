import numpy as np
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from keras.utils import np_utils

# set up your dataset folder.
dataset_folder = './data/data2/'

# data preprocessing
data = pd.read_csv(dataset_folder + 'train.csv')
sentences_pre = []
for sentence in data['text']:
    sentences_pre.append(gensim.parsing.preprocess_string(sentence))
dct = gensim.corpora.Dictionary(sentences_pre)
dct.filter_extremes(no_below=20, no_above=0.3)

# X
bow_len = len(dct.token2id.keys())

def bow2vec(bow):
    output = np.zeros(bow_len)
    for word in bow:
        output[word[0]] = word[1]
    return output

X = np.array([bow2vec(dct.doc2bow(sentence)) for sentence in sentences_pre])

# Y
le = preprocessing.LabelEncoder()
le.fit(data['author'])
Y = le.transform(data['author'])
Y = np_utils.to_categorical(Y)

# train ANN
model_sk = MLPClassifier()
scores = cross_val_score(model_sk, X, Y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))