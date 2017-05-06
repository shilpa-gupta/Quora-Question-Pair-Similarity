import pandas as pd
import gensim.models
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt


"""
This function is to train a word2vec model based on skipgram approach
using the Quora Data itself as vocabulary
"""
def gen_w2vmodel(filename):
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

"""
This function is used to load the glove embeddings into a dictionary
which we will later use to generate the embedding representation of questions
"""
def load_dict():
    model = {}
    filePath = "../data/glove.6B.50d.txt"
    with open(filePath, 'r', encoding='utf-8') as input:
        count = 0
        for line in input:
            count += 1
            print(count)
            tokens = line.split(" ")
            tokens[len(tokens)-1] = tokens[len(tokens)-1].strip()
            word = tokens[0]
            embed = np.array([float(x) for x in tokens[1:]])
            model[word] = embed
    return model

#load_dict()

"""
This function is for calculating distances between the question pairs 
and saving them in a csv file.
"""
def calculate_dists(filename):
    model = load_dict()
    count = 0  
    data_frame = pd.read_csv(filename)
    with open("../data/distances_glove.csv", 'a') as dist_file:
        for index, row in data_frame.iterrows():
            print(count)
            count += 1
            q1 = row['question1']
            q2 = row['question2']
            true_label = int(row['is_duplicate'])

            # python 2+
            # q1_word = word_tokenize(q1.lower().decode('utf-8'))
            # q2_word = word_tokenize(q2.lower().decode('utf-8'))

            # python 3+
            q1_word = word_tokenize(q1.lower())
            q2_word = word_tokenize(q2.lower())

            # For model generated embeddings
            # sum_q1 = np.zeros(100, )
            # sum_q2 = np.zeros(100, )

            # For glove embeddings
            sum_q1 = np.zeros(50, )
            sum_q2 = np.zeros(50, )
            n = 0
            for word in q1_word:
                if word in model.keys():
                    n += 1
                    sum_q1 += model[word]
            if n != 0:
                sum_q1 = sum_q1 / n

            n = 0
            for word in q2_word:
                if word in model.keys():
                    n += 1
                    sum_q2 += model[word]
            if n != 0:
                sum_q2 = sum_q2 / n

            cos_distance = spatial.distance.cosine(sum_q1, sum_q2)
            eucledian_distance = spatial.distance.euclidean(sum_q1, sum_q2)
            dist_file.write(",".join(str(x) for x in [true_label, cos_distance, eucledian_distance]) + '\n')
#calculate_dists("../data/QuoraData.csv")

"""
this function is to visulise the scatter plot of distances among the question pairs
"""
def vis_distances(distFile):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    with open(distFile, 'r') as inFile:
        count = 0;
        for line in inFile:
            if count == 0:
                count = 1
                continue
            row = line.split(",")
            if int(row[0]) == 0:
                x1.append(float(row[1]))
                y1.append(float(row[2].strip()))
            else:
                x2.append(float(row[1]))
                y2.append(float(row[2].strip()))
    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')
    plt.show()

#vis_distances("../data/distances.csv")


"""
Here we are calculating the accuracy after coming up the threshoulds according to
scatter plots plotted above

Threshoulds
normal embeddings :
    cosine distance (x-axis in scatter) (> 0.251566 are red)
    euclidian distance (x-axis in scatter) ( > 2.063 are red)

glove embeddings : 
    cosine distance (x-axis in scatter) (>0.291009 are red)
    euclidian distance (x-axis in scatter) (>2.58611 are red)
"""
def calculate_accuracy(distFile):
    threshold_cos = 0.291009
    threshold_euc = 2.58611
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(distFile, 'r') as dist_file:
        count = 0
        for line in dist_file:
            if count == 0:
                count += 1
                continue
            cols = line.split(",")
            # cos_distance = cols[1]
            euc_distance = cols[2].strip()
            true_label = cols[0]
            if float(euc_distance) > threshold_euc:
                if int(true_label) == 0:
                    tn += 1
                else:
                    fn += 1
            else:
                if int(true_label) == 1:
                    tp += 1
                else:
                    fp += 1
    print("glove embeddings for cosine distance")
    print(" true positive :", tp)
    print(" true negative :", tn)
    print(" flase positive :", fp)
    print(" false negative :", fn)

calculate_accuracy("../data/distances_glove.csv")