import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')

model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin.gz', binary=True)
"""
Basic fetaures :
    Length of question1
    Length of question2
    Difference in the two lengths
    Character length of question1 without spaces
    Character length of question2 without spaces
    Number of words in question1
    Number of words in question2
    Number of common words in question1 and question2
"""
def length_features(data):	
	data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
	data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
	data['diff_len'] = data.len_q1 - data.len_q2
	data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
	data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
	

def fuzzy_features(data):
	data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)	
	data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)	
	data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)	

def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def gen_distance_metrics(data,question1_vectors,question2_vectors):

	data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
	                                                          np.nan_to_num(question2_vectors))]
	data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
	data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
	data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
	data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


def main():
	data = pd.read_csv('../data/QuoraData.csv')
	labels=data['is_duplicate']
	data.drop(['is_duplicate','qid1','qid2'],axis=1,inplace=True)
	
	length_features(data)
	fuzzy_features(data)
	
	question1_vectors = np.zeros((data.shape[0], 300))
	error_count = 0

	for i, q in tqdm(enumerate(data.question1.values)):
		question1_vectors[i, :] = sent2vec(q)
	question2_vectors  = np.zeros((data.shape[0], 300))
	for i, q in tqdm(enumerate(data.question2.values)):
		question2_vectors[i, :] = sent2vec(q)
	gen_distance_metrics(data,question1_vectors,question2_vectors)

	data['labels']=labels
	data.to_csv('../data/quora_features.csv', index=False)
	
def modify_csv(filename):
	data = pd.read_csv(filename)
	data.drop('\xef\xbb\xbfid',axis=1,inplace=True)
	data.to_csv('../data/quora_features.csv', index=False)



def writeTrainDataToARFF():	
	with open('../data/quora' + ".arff", 'w') as f:
		data = pd.read_csv('../data/quora_features.csv')
		featureNames = data.columns
		f.write('@RELATION ' + 'quora' + '\n')
		for fn in featureNames:
			f.write('@ATTRIBUTE ' + fn + ' REAL\n')
		f.write('@ATTRIBUTE class {0,1}'+'\n')
		f.write('@DATA\n')
		with open('../data/quora_features','r') as fin:
			for line in fin:
				line=line.strip()
				f.write(str(line)+'\n')
	f.close()
writeTrainDataToARFF()