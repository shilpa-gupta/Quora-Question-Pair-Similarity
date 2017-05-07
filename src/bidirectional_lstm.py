import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, merge, Input, Bidirectional 
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import cPickle

logger_file = open('logfile_asmita', 'w')
# Create data matrices and labels list from processed data tuples
def create_data_matrices(input_dataset):
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

# load the processed Quora dataset
with open("../data/data_tuples_glovem.p", "rb") as f:
	total_dataset = cPickle.load(f)
print("Loaded the data tuples")
logger_file.write("Loaded the data tuples")
dataset = []

#pre_data_tuples of the form ('what is the story of kohinoor koh i noor diamond', 'what would happen if the indian government stole the kohinoor koh i noor diamond back', 0)

for instance in total_dataset:
	len_s1 = len(instance[0].split())
	len_s2 = len(instance[1].split())
	if(len_s1==0 or len_s2==0):
		continue
	dataset.append(instance)
print("Removed pairs with empty sentences. Remaining num. of data tuples ", len(dataset))
logger_file.write("Removed pairs with empty sentences. Remaining num. of data tuples " + str(len(dataset)))
# Load glove vector dict (only for the needed words)
with open("../data/needed_glovem_dict.p", "rb") as f:
	glove_embedding_dict = cPickle.load(f)

print("Loaded the Glove dictionary for necessary words")
logger_file.write("Loaded the Glove dictionary for necessary words")
glove_dim = glove_embedding_dict[glove_embedding_dict.keys()[1]].shape[0]
vocab_size = 80405 + 1#80313#80419 # Pass this from analyze_data, instead of hardcoding.80312

# Initialize embedding matrix with each entry sampled uniformly at random between -1.0 and 1.0
init_glove_matrix =  np.random.uniform(-1.0, 1.0, size=(vocab_size, glove_dim))
print("Initialized glove matrix with uniform. Will overwrite known vectors in it now")
logger_file.write("Initialized glove matrix with uniform. Will overwrite known vectors in it now")
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
				init_glove_matrix[unique_id] = glove_embedding_dict[word]
			unique_id += 1

	for word in sentence2:
		if(word not in word_to_id):
			word_to_id[word] = unique_id
			if word in glove_embedding_dict:
				init_glove_matrix[unique_id] = glove_embedding_dict[word]
			unique_id += 1

print("Max sentence length in data ", sentence_word_limit)
logger_file.write("Max sentence length in data " + str(sentence_word_limit))
sentence_lengths = np.array(sentence_lengths)
print("Num more than 50 ", np.sum(sentence_lengths>=50))
#logger_file.write("Num more than 50 ", np.sum(sentence_lengths>=50))
print("Num more than 60 ", np.sum(sentence_lengths>=60))

#logger_file.write("Num more than 60 ", np.sum(sentence_lengths>=60))
TUNING_PARAMETER = 60
sentence_word_limit = min(sentence_word_limit, TUNING_PARAMETER)


# Train, Test lists creation. Test here is technically more like Validation
train_data_matrix1 = []
train_data_matrix2 = []
label_train = []
test_data_matrix1 = []
test_data_matrix2 = []
label_test = []

train_pc = 0.8
num_train = int(np.ceil(train_pc*len(dataset)))
random.seed(186) # Fixing random seed for reproducibility
random.shuffle(dataset)

# TRAIN - TEST SPLIT OF THE TUPLES
train_dataset = dataset[0: num_train]
test_dataset = dataset[num_train:]
print("Num of training examples ", len(train_dataset))
logger_file.write("Num of training examples "+ str(len(train_dataset)))

train_data_matrix1, train_data_matrix2, label_train = create_data_matrices(train_dataset)
test_data_matrix1, test_data_matrix2, label_test = create_data_matrices(test_dataset)
print("Created Training and Test Matrices, and corresponding label vectors")
logger_file.write("Created Training and Test Matrices, and corresponding label vectors")

# create the model
embedding_size= 300
#vocab_size = total_num_words + 1 # since the <none> token is extra
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[init_glove_matrix]))
model.add(Bidirectional(LSTM(100, dropout_W=0.5, dropout_U=0.5)))
print("Done building core model")
logger_file.write("Done building core model")

# Inputs to Full Model
#input_dim = sentence_word_limit
sentence_input1 = Input(shape=(sentence_word_limit,))
sentence_input2 = Input(shape=(sentence_word_limit,))

# Send them through same model (weights will be thus shared)
processed_1 = model(sentence_input1)
processed_2 = model(sentence_input2)

print("Going to merge the two branches at model level")
logger_file.write("Going to merge the two branches at model level")

merged = merge([processed_1, processed_2], mode='concat')

# Add an FC layer before the Clf layer (non-lin layer after the lstm 'thought vecs' concatenation)
merged_fc = Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001), name='merged_fc')(merged)
merged_fc_drop = Dropout(0.4)(merged_fc) # Prevent overfitting at the fc layer

main_output = Dense(1, activation='sigmoid', name='main_output')(merged_fc_drop)

full_model = Model( input=[sentence_input1, sentence_input2], output=main_output )

full_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(full_model.summary())
#logger_file.write(full_model.summary())

#saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath="../models/weights-{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_acc', verbose=1, save_best_only=False)

full_model.fit( [train_data_matrix1, train_data_matrix2], label_train, validation_data=([test_data_matrix1, test_data_matrix2], label_test), nb_epoch=12, batch_size=128, verbose=1, callbacks=[checkpointer])




# Final evaluation of the model
scores = full_model.evaluate( [test_data_matrix1, test_data_matrix2], label_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
logger_file.write("Accuracy"  + str(scores[1]*100))
logger_file.write("DUMPING DATA")
