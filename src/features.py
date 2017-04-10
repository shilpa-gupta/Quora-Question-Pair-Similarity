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

model = gensim.models.Word2Vec.load('../models/embeddings')
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

def word_mover_distance(s1,s2):
	s1 = str(s1).lower().split()
	s2 = str(s2).lower().split()
	stop_words = stopwords.words('english')
	s1 = [w for w in s1 if w not in stop_words]
	s2 = [w for w in s2 if w not in stop_words]	
	return model.wmdistance(s1, s2)

def gen_distance_metrics

def main():
	data = pd.read_csv('../sample/quora_features_local.csv')
	#length_features(data)
	#fuzzy_features(data)

	data['wmd'] = data.apply(lambda x: word_mover_distance(x['question1'], x['question2']), axis=1)
	


	data.to_csv('../sample/quora_features_local.csv', index=False)



main()