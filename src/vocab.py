import pandas as pd
import gensim.models
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np

"""
TO-DO till tuesday
1. gen embeddings --> find cosine/euclidean distance --> check the accuracy
2. gen embeddings --> add the 2 questions embeddings --> classify
3. try appending not avg and both the above approch
4. use standard/glove embeddings and try above 3 approches
"""

def gen_vocab(filename):
    count = 0
    data_frame = pd.read_csv(filename)
    vocab = []
    for index, row in data_frame.iterrows():
        print(count)
        count += 1
        words = []
        q1 = row['question1']
        q2 = row['question2']

        # python 2+
        # words.append(word_tokenize(q1.lower().decode('utf-8')))
        # words.append(word_tokenize(q2.lower().decode('utf-8')))

        # python 3+
        words.append(word_tokenize(q1.lower()))
        words.append(word_tokenize(q2.lower()))
        for word in words:
            if word not in vocab:
                vocab.append(word)

    model = gensim.models.Word2Vec(vocab, size=100, min_count=1, sg=1)
    w2v = dict(zip(model.index2word, model.syn0))
    model.save('embeddings')

#
def gen_features(filename):
    model = gensim.models.Word2Vec.load('../models/embeddings')
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    count = 0
    data_frame = pd.read_csv(filename)
    for index, row in data_frame.iterrows():
        # print(count)
        count += 1
        q1 = row['question1']
        q2 = row['question2']
        true_label = int(row['is_duplicate'])

        # python 2+
        # q1_word = word_tokenize(q1.lower().decode('utf-8'))
        # q2_word = word_tokenize(q2.lower().decode('utf-8'))

        #python 3+
        q1_word = word_tokenize(q1.lower())
        q2_word = word_tokenize(q2.lower())

        sum_q1 = np.zeros(100, )
        sum_q2 = np.zeros(100, )
        n = 0
        for word in q1_word:
            if word in model:
                n += 1
                sum_q1 += model[word]
        sum_q1 = sum_q1 / n
        n = 0
        for word in q2_word:
            if word in model:
                n += 1
                sum_q2 += model[word]
        sum_q2 = sum_q2 / n
        cos_distance = spatial.distance.cosine(sum_q1, sum_q2)
        eucledian_distance = spatial.distance.euclidean(sum_q1, sum_q2)
        print("************************************")
        print(true_label)
        print(cos_distance)
        print(eucledian_distance)
        threshold_cos = 0
        threshold_euc = 0
        if cos_distance > threshold_cos:
            if true_label == 0:
                tn += 1
            else:
                fn += 1
        else:
            if true_label == 1:
                tp += 1
            else:
                fp += 1
    print(" true positive :", tp)
    print(" true negative :", tn)
    print(" flase positive :", fp)
    print(" false negative :", fn)


gen_features("../data/QuoraData.csv")
