import h5py
import numpy as np
file_name = "../models/weights-11-0.49.hdf5"
final_built_model = h5py.File(file_name,'r+')
import numpy as np
import random
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, merge, Input, Bidirectional 
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
#from keras.np_utils import probas_to_classes 
import cPickle
final_built_model = load_model(file_name)

logger_file = open('logfile', 'w')

def generate_matrix(input_dataset):
	question_array1 = []
	question_array2 = []
	is_same_array = []
	for instance in input_dataset:
		sentence1 = (instance[0]).split()
		sentence2 = (instance[1]).split()
		is_same_question = instance[2]
		sentence1_id_array = [0]*sentence_word_limit
		sentence2_id_array = [0]*sentence_word_limit
		i = 0
		for word in sentence1:
			if((i+1) == sentence_word_limit):
				break
			sentence1_id_array[i] = word_to_id[word]
			i += 1
		i = 0
		for word in sentence2:
			if((i+1) == sentence_word_limit):
				break
			sentence2_id_array[i] = word_to_id[word]
			i += 1

		question_array1.append(np.array(sentence1_id_array))
		question_array2.append(np.array(sentence2_id_array))
		is_same_array.append(is_same_question)

	return np.array(question_array1), np.array(question_array2), is_same_array


with open("../data/data_tuples_glovem_orj.p", "rb") as f:
	total_dataset = cPickle.load(f)

logger_file.write("LOAD DATA INSTANCES")
dataset = []
#total_dataset = total_dataset[1:10]
#pre_data_tuples of the form ('what is the story of kohinoor koh i noor diamond', 'what would happen if the indian government stole the kohinoor koh i noor diamond back', 0)

for instance in total_dataset:
	len_s1 = len(instance[0].split())
	len_s2 = len(instance[1].split())
	if(len_s1==0 or len_s2==0):
		continue
	dataset.append(instance)
print("REMOVE EMPTY SENTENCES, REMAINING INSTANCES", len(dataset))
logger_file.write("REMOVE EMPTY SENTENCES, REMAINING INSTANCES" + str(len(dataset)))
# Load glove vector dict (only for the needed words)
with open("../data/needed_glovem_dict.p", "rb") as f:
	glove_embedding_dict = cPickle.load(f)

print("AVAILABLE GLOVE DICTIONARY LOADED")
logger_file.write("AVAILABLE GLOVE DICTIONARY LOADED")
glove_dim = glove_embedding_dict[glove_embedding_dict.keys()[1]].shape[0]
vocab_size = 80405 + 1#80313#80419 # Pass this from analyze_data, instead of hardcoding.80312

# Initialize embedding matrix with each entry sampled uniformly at random between -1.0 and 1.0
precomputed_glove_embeddings =  np.random.uniform(-1.0, 1.0, size=(vocab_size, glove_dim))
print("RANDOM EMBEDDING ASSIGNMENT TO WORDS")
logger_file.write("RANDOM EMBEDDING ASSIGNMENT TO WORDS")
# First create a dictionary from word to idx (for all distinct words)
word_to_id = {}
sentence_word_limit = 0
sentence_lengths = []
unique_id = 1 # Start with 1, since 0 is used for <none> token (i.e., padding sentences to get to max length)
words_in_order = []
for instance in dataset:
	sentence1 = instance[0].split()
	sentence2 = instance[1].split()

	# Update max_sentence_len as necessary
	sentence_word_limit = max(sentence_word_limit, len(sentence1))
	sentence_word_limit = max(sentence_word_limit, len(sentence2))
	sentence_lengths.append(len(sentence1))
	sentence_lengths.append(len(sentence2))

	for word in sentence1:
		if(word not in word_to_id):
			word_to_id[word] = unique_id
			if(word in glove_embedding_dict):
				precomputed_glove_embeddings[unique_id] = glove_embedding_dict[word]
			unique_id += 1

	for word in sentence2:
		if(word not in word_to_id):
			word_to_id[word] = unique_id
			if word in glove_embedding_dict:
				precomputed_glove_embeddings[unique_id] = glove_embedding_dict[word]
			unique_id += 1

print("MAX WORD LIMIT ", sentence_word_limit)
logger_file.write("MAX WORD LIMIT " + str(sentence_word_limit))
sentence_lengths = np.array(sentence_lengths)

TUNING_PARAMETER = 60
sentence_word_limit = 60
#sentence_word_limit = min(sentence_word_limit, TUNING_PARAMETER)

train_data_matrix1 = []
train_data_matrix2 = []
label_train = []
test_data_matrix1 = []
test_data_matrix2 = []
label_test = []

train_pc = 0.8
num_train = int(np.ceil(train_pc*len(dataset)))
random.seed(186)
random.shuffle(dataset)

#train_dataset = dataset[0: num_train]
test_dataset =  dataset[num_train: num_train+200]

#print("TRAIN DATASET LENGTH", len(train_dataset))
#logger_file.write("TRAIN DATASET LENGTH "+ str(len(train_dataset)))

#train_data_matrix1, train_data_matrix2, label_train = generate_matrix(train_dataset)
test_data_matrix1, test_data_matrix2, label_test = generate_matrix(test_dataset)
print("GENERATING TRAINING AND TEST MATRIX")
logger_file.write("GENERATING TRAINING AND TEST MATRIX")
#print(test_data_matrix1)
#embedding_size= 300
#model = Sequential()
#model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[precomputed_glove_embeddings]))
#model.add(Bidirectional(LSTM(100, dropout_W=0.5, dropout_U=0.5)))

#sentence_input1 = Input(shape=(sentence_word_limit,))
#sentence_input2 = Input(shape=(sentence_word_limit,))

#merge_input1 = model(sentence_input1)
#merge_input2 = model(sentence_input2)

#merged = merge([model(sentence_input1), model(sentence_input2)], mode='concat')

#fully_connected = Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001), name='fully_connected')(merged)
#fully_connected_drop = Dropout(0.4)(fully_connected) 

#final_layer = Dense(1, activation='sigmoid', name='final_layer')(fully_connected_drop)

#final_built_model = Model( input=[sentence_input1, sentence_input2], output=final_layer )

#final_built_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(final_built_model.summary())

#model_writer = ModelCheckpoint(filepath="../models/model-{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_acc', verbose=1, save_best_only=False)

#final_built_model.fit( [train_data_matrix1, train_data_matrix2], label_train, validation_data=([test_data_matrix1, test_data_matrix2], label_test), nb_epoch=2, batch_size=128, verbose=1, callbacks=[model_writer])
scores = final_built_model.predict( [test_data_matrix1, test_data_matrix2], verbose=1)
i = 0
print(scores)
for instances in test_dataset:
	if(scores[i] > 0.5):
		print(instances, "DUPLICATES")
	else:
		print(instances, "DIFFERENT")
	i += 1

scores = final_built_model.evaluate( [test_data_matrix1, test_data_matrix2], label_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
logger_file.write("Accuracy"  + str(scores[1]*100))
logger_file.write("DUMPING DATA")
