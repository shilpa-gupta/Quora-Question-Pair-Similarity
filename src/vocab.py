import pandas as pd
import pickle
import gensim.models
from nltk.tokenize import word_tokenize
from scipy import spatial

def gen_vocab(filename):
	count=0
	data_frame = pd.read_csv(filename)
	vocab=[]
	for index,row in data_frame.iterrows():
		print(count)
		count+=1
		words = []		
		q1 =row['question1']
		q2 = row['question2']
		words.append(word_tokenize(q1.lower().decode('utf-8')))
		words.append(word_tokenize(q2.lower().decode('utf-8')))
		for word in words:
			if word not in vocab:
				vocab.append(word)

	model = gensim.models.Word2Vec(vocab, size=100, min_count=1,sg=1)
	w2v = dict(zip(model.index2word, model.syn0))
	model.save('embeddings')

def gen_features(filename):
	model = gensim.models.Word2Vec.load('embeddings')
	tp=0
	fp=0
	tn=0
	fn=0
	data_frame = pd.read_csv(filename)	
	for index,row in data_frame.iterrows():
		print(count)
		count+=1
		q1 =row['question1']
		q2 = row['question2']
		true_label = row['is_duplicate']
		q1_word = word_tokenize(q1.lower().decode('utf-8'))
		q2_word = word_tokenize(q2.lower().decode('utf-8'))
		sum_q1 = np.zeros(100,)
		sum_q2 = np.zeros(100,)
		for word in q1_word:
			if word in model:
				count+=1
				sum_q1 += model[word]
		for word in q1_word:
			if word in model:
				count+=1
				sum_q2 += model[word]
		cos_distance = spatial.distance.cosine(sum_q1,sum_q2)
		eucledian_distance = spatial.distance.eucledian(sum_q1,sum_q2)
		print("************************************")
		print(true_label)
		print(cos_distance)
		print(eucledian_distance)
		threshold_cos =0
		threshold_euc =0
		if cos_distance > threshold_cos:
			if true_label==0:
				tn +=1
			else:
				fn+=1
		else:
			if true_label==1:
				tp +=1
			else:
				fp+=1
	print(" true positive :" ,tp)
	print(" true negative :" ,tn)
	print(" flase positive :" ,fp)
	print(" false negative :" ,fn)

gen_vocab("/home/nidhi/NLP_Project/data/QuoraData.csv")