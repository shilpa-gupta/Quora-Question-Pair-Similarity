import numpy as np
import csv, json
from os.path import exists
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2

from keras import backend as K
from keras.utils.data_utils import get_file


'''initial length constants'''
LEN_EMBEDDING = 300
MAX_SENTENCE_LENGTH = 25
NB_LIMIT = 200000


if exists('../data/question1_train.npy') and exists('../data/question2_train.npy') and exists('../data/true_labels_train.npy') and exists('../data/nb_words.json') and exists('../data/word_embedding_matrix.npy'):
    q1_data = np.load(open('../data/question1_train.npy', 'rb'))
    q2_data = np.load(open('../data/question2_train.npy', 'rb'))
    labels = np.load(open('../data/true_labels_train.npy', 'rb'))
    word_embeddings = np.load(open('../data/word_embedding_matrix.npy', 'rb'))
    with open('../data/nb_words.json', 'r') as f:
        nb_words = json.load(f)['nb_words']

else:
    q_instance_1 = []
    q_instance_2 = []
    is_duplicate = []
    with open('../data/QuoraData.csv') as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            q_instance_1.append(row['question1'])
            q_instance_2.append(row['question2'])
            is_duplicate.append(row['is_duplicate'])

    questions = q_instance_1 + q_instance_2
    tokenizer = Tokenizer(nb_words='../data/nb_words.json')
    tokenizer.fit_on_texts(questions)
    seq_1 = tokenizer.texts_to_sequences(q_instance_1)
    seq_2 = tokenizer.texts_to_sequences(q_instance_2)
    word_index = tokenizer.word_index


    embeddings_index = {}
    with open('../data/glove.840B.300d.txt') as infile:
        for line in infile:
            values = line.split(' ')
            word = values[0]
	    embeddings_index[word] = np.asarray(values[1:], dtype='float32')



    nb_words = min(NB_LIMIT, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, LEN_EMBEDDING))
    for word, i in word_index.items():
        if i > NB_LIMIT:
            continue
        if embeddings_index.get(word) is not None:
            word_embedding_matrix[i] = embeddings_index.get(word)
        
    

    q1_data = pad_sequences(seq_1, maxlen=MAX_SENTENCE_LENGTH)
    q2_data = pad_sequences(seq_2, maxlen=MAX_SENTENCE_LENGTH)
    labels = np.array(is_duplicate, dtype=int)
    
    np.save(open('../data/question1_train.npy', 'wb'), q1_data)
    np.save(open('../data/question2_train.npy', 'wb'), q2_data)
    np.save(open('../data/true_labels_train.npy', 'wb'), labels)
    np.save(open('../data/word_embedding_matrix.npy', 'wb'), word_embedding_matrix)
    with open('../data/nb_words.json', 'w') as f:
        json.dump({'nb_words': nb_words}, f)


#Processing the questions and splitting into test and train
data = np.stack((q1_data, q2_data), axis=1)
true_w = labels
X_train, X_test, y_train, y_test = train_test_split(data, true_w, test_size=0.1, random_state=13371447)
ques1_train = X_train[:,0]
ques2_train = X_train[:,1]
ques1_test = X_test[:,0]
ques2_test = X_test[:,1]
'''
Embedding model for both the questions
Time distributed lstm layer over the embeddings with relu activation function
'''
ques1 = Sequential()
ques1.add(Embedding(nb_words + 1, LEN_EMBEDDING, weights=[word_embedding_matrix], input_length=MAX_SENTENCE_LENGTH, trainable=False))
ques1.add(TimeDistributed(Dense(LEN_EMBEDDING, activation='relu')))
ques1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(LEN_EMBEDDING, )))
ques2 = Sequential()
ques2.add(Embedding(nb_words + 1, LEN_EMBEDDING, weights=[word_embedding_matrix], input_length=MAX_SENTENCE_LENGTH, trainable=False))
ques2.add(TimeDistributed(Dense(LEN_EMBEDDING, activation='relu')))
ques2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(LEN_EMBEDDING, )))
#
model = Sequential()
model.add(Merge([ques1, ques2], mode='concat'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
#classifier layer on relu
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])

call_backs = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
'''generating the model for 25 epochs with 0.1 validation split'''
history = model.fit([ques1_train, ques2_train], 
                    y_train, 
                    nb_epoch=25, 
                    validation_split=0.1, 
                    verbose=1, 
                    callbacks=call_backs)
model.load_weights(MODEL_WEIGHTS_FILE)

'''Evaluation section on test data'''
loss, accuracy, precision, recall, fbeta_score = model.evaluate([ques1_test, ques2_test], y_test)
print('loss      = ',loss)
print('accuracy  = ',accuracy)
print('precision = ',precision)
print('recall    = ',recall)
print('F         = ',fbeta_score)
