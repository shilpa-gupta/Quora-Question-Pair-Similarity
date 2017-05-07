"""
In this approach we are Extracting few features from the question pairs
and generating the feature vectors in an arff file, which we used to run 
SVM classifier using weka tool
"""

import cPickle
import pandas as pd
import numpy as np
import gensim
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean
from nltk import word_tokenize

stopWords = stopwords.words('english')

model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
"""
Basic features :
	Length of Question 1
	Length of Question 2
	Difference between 2 Lengths
	Character Length of Question 1 without spaces
	Character Length of Question 2 without spaces
	Number of words in Question 1
	Number of words in Question 2
	Number of comman words in Question 1 and Question 2
"""

def basic_feat(data):
	data['q1Len'] = data.question1.apply(lambda x : len(str(x)))
	data['q2Len'] = data.question2.apply(lambda x : len(str(x)))
	data['len_diff'] = data.q1Len - data.q2Len
	data['q1CharLen'] = data.question1.apply(lambda x : len(''.join(set(str(x).replace(' ','')))))
	data['q2CharLen'] = data.question2.apply(lambda x : len(''.join(set(str(x).replace(' ','')))))
	data['q1WordLen'] = data.question1.apply(lambda x : lem(str(x).split()))
	data['q2WordLen'] = data.question2.apply(lambda x : len(str(x).split()))
	data['commonWords'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

"""
This is the function to calculate Word mover distance 
between 2 sentences
"""
def wmd(s1, s2):
	s1 = str(s1).lower().split()
	s2 = str(s2).lower().split()
	stopWords = stopwords.words('english')
	s1 = [w for w in s1 if w not in stopwords]
	s2 = [w for w in s2 if w not in stopwords]
	return model.wmdistance(s1, s2)

"""
This is the function to calculate normalized word mover distance
where we are using the normalized word2vec model created using google news corpora
"""
def norm_wmd(s1, s2):
	s1 = str(s1).lower().split()
	s2 = str(s2).lower().split()
	stopWords = stopwords.words('english')
	s1 = [w for w in s1 if w not in stopWords]
	s2 = [w for w in s2 if w not in stopWords]
	return norm_model.wmdistance(s1, s2)

"""
this is the method to convert the questions from text to their 
vector representation by averaging the word embeddings
"""
def sent2vec(s):
	words = str(s).lower().decode('utf-8')
	words = word_tokenize(words)
	words = [w for w in words if not w in stopWords]
	words = [w for w in words if w.isalpha()]
	vec = []
	for w in words:
		try:
			vec.append(model[w])
		except:
			continue
	vec = np.array(vec)
	total = vec.sum(axis=0)
	return total/np.sqrt((v**2).sum()) 


"""
this method is to calculate distance related features 
"""
def distance_feat(data, q1_vec, q2_vec):
	data['wmd'] = data.apply(lambda x : wmd(x['question1'], x['question2']), axis=1)
	data['norm_wmd'] = data.apply(lambda x : norm_wmd(x['question1'], x['question2']), axis=1)
	data['cosineDist'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
	data['euclideanDist'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(q1_vec),np.nan_to_num(q2_vec))]

def main():
	data = pd.load_csv('../data/QuoraData.csv')
	labels = data['is_duplicate']
	data.drop(['is_duplicate', 'qid1', 'qid2'], axis=1, inplace=True)

	basic_feat(data)

	q1_vecs = np.zeros((data.shape[0], 300))
	error = 0

	for i, q in tqdm(enumerate(data.question1.values)):
		question1_vectors[i, :] = sent2vec(q)
	question2_vectors  = np.zeros((data.shape[0], 300))
	for i, q in tqdm(enumerate(data.question2.values)):
		question2_vectors[i, :] = sent2vec(q)
	distance_feat(data,question1_vectors,question2_vectors)

	data['labels'] = labels
	data.to_csv('../data/quora_feat.csv', index=False)


def DataToArff():
	with open('../data/quora'+".arff", 'w') as f:
		data = pd.read_csv('../data/quora_feat.csv')
		featNames = data.columns
		featureNames = data.columns
		f.write('@RELATION ' + 'quora' + '\n')
		for fn in featureNames:
			f.write('@ATTRIBUTE ' + fn + ' REAL\n')
		f.write('@ATTRIBUTE class {0,1}'+'\n')
		f.write('@DATA\n')
		with open('../data/quora_feat.csv','r') as fin:
			for line in fin:
				line=line.strip()
				f.write(str(line)+'\n')
	f.close()

DataToArff()