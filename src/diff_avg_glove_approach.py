"""
In this approach we are first converting the question pair
to feature vectors using their embedding representations
"""

import pickle
import sys
import re
import spell
from spell import correction
import csv
import numpy as np
from string import ascii_lowercase as ascii_l
import random

# load Quora Question pair data in a list
def load_data(infile):
	rows = []
	with open(infile, 'r', encoding='utf-8') as input:
		input = csv.reader(input, delimiter=',')
		for index, row in enumerate(input):
			if index > 0:
				rows.append(row)
	return rows

# saving the Glove embeddings words into a diction for fast search
def glove_dict(infile):
	dict = {}
	with open(infile, 'r', encoding='utf-8') as input:
		for line in input:
			word = line.split(" ")[0]
			dict[word] = 1

	with open("../data/glove_dict.p", "wb") as f:
		pickle.dump(dict, f)

# This method is to preprocess the Question text
# Here we are removing all unnecessary characters from the sentences
def preprocessing(text):
	text = text.strip(" ?")
	text = text.replace("/","")
	text = text.replace("-","")
	text = text.replace("(","")
	text = text.replace(")","")
	text = text.replace(".","")
	text = text.replace("?","")
	text = text.replace(",","")
	text = text.replace(";","")
	text.strip()
	text = re.sub(r'[^a-zA-Z0-9]','',text)
	return text

def process_data():
	tokens_n = 0
	all_wordDict = {}
	found_cnt = 0
	data_tuples = []
	rows = load_data("../data/QuoraData.csv")
	with open("../data/glove_dict.p","rb") as f:
		glove_dict = pickle.load(f)

	for index, row in enumerate(rows):
		q1 = preprocessing(row[3])
		words1 = q1.split()
		final_words1 = []

		q2 = preprocessing(row[4])
		words2 = q2.split()
		final_words2 = []

		for word in words1:
			tokens_n += 1
			word = word.lower()
			if word not in all_wordDict:
				if word in glove_dict:
					found_cnt += 1
				else :
					all_wordDict[word] = 1
			final_words1.append(word)
		for word in words2:
			tokens_n += 1
			word = word.lower()
			if word not in all_wordDict:
				if word in glove_dict:
					found_cnt += 1
				else:
					all_wordDict[word] = 1
			final_words2.append(word)

		processed_q1 = " ".join(final_words1)
		processed_q2 = " ".join(final_words2)

		curr_label = int(row[5].strip())
		data_tuples.append((processed_q1, processed_q2, curr_label))
	print("number of data points", len(data_tuples))

	with open("../data/data_tuple_gloveem.p",'wb') as f:
		pickle.dump(data_tuples, f)

def generate_embeddigns():
	with open("../data/data_tuple_gloveem.p", "rb") as f:
		data_tuples = pickle.load(f)
	needed_words = {}
	for tup in data_tuples:
		s1 = tup[0]
		s2 = tup[1]
		for word in s1.split():
			if word not in needed_words:
				needed_words[word] = 1
		for word in s2.split():
			if word not in needed_words:
				needed_words[word] = 1
	needed_glove_dict = {}
	vec_file = open("../data/glove.42B.300d.txt", encoding="utf-8")
	for line in vec_file:
		words = line.split()
		if words[0] in needed_words:
			needed_glove_dict[words[0]] = np.asarray(words[1:])

	vec_file.close()

	with open("../data/needed_glove_dict.p", "wb") as f:
		pickle.dump(needed_glove_dict,f)  


def gen_features():
    with open("../data/data_tuple_gloveem.p", "rb") as f:
        data_tuples = pickle.load(f)

    with open("../data/needed_glove_dict.p", "rb") as f:
        glove_dict = pickle.load(f)

    chars = []
    for ch in ascii_l:
        all_chars.append(ch)

    for idx in range(10):
        all_chars.append(str(idx))

    char_to_idx = {ch: ix for ix, ch in enumerate(all_chars)}
    num_examples = len(data_tuples)
    glove_dim = 300
    boc_dim = 36
    curr_idx = 0
    feat_dim = glove_dim * 2 + boc_dim * 2  
    feat_matrix = np.zeros((num_examples, feat_dim))
    labels = np.zeros(num_examples)
    with open("../data/features_embed_append.csv", 'w', encoding="utf-8") as csv_file:
        for idx, curr_tuple in enumerate(data_tuples):
            if (idx + 1) % 1000 == 0:
                print("Processing example ", idx + 1)
            sent1 = curr_tuple[0]
            sent2 = curr_tuple[1]
            sent1 = sent1.strip()
            sent2 = sent2.strip()

            if(len(sent1.split()) == 0 or len(sent2.split()) == 0):
                continue

            boc1 = np.zeros(36)
            vec1 = np.zeros(300)
            denom = 0
            for word in sent1.split():
                if word in glove_dict:
                    curr_vec = glove_dict[word]
                    denom += 1
                else:
                    for ch in word:
                        boc1[char_to_idx[ch]] += 1
                    continue
                vec1 += np.array(curr_vec, dtype=np.float)
            if denom != 0:
                vec1 /= denom

            boc2 = np.zeros(36)
            vec2 = np.zeros(300)
            denom = 0
            for word in sent2.split():
                if word in glove_dict:
                    curr_vec = glove_dict[word]
                    denom += 1
                else:
                    for ch in word:
                        boc2[char_to_idx[ch]] += 1
                    continue
                vec2 += np.array(curr_vec, dtype=np.float)
            if denom != 0:
                vec2 /=denom
            feat_matrix[curr_idx] = np.hstack((vec1, vec2, boc1, boc2))
            labels[curr_idx] = curr_tuple[2]
            curr_idx += 1
            feat_vec = []
            feat_vec.extend(vec1.tolist())
            feat_vec.extend(vec2.tolist())
            feat_vec.extend(boc1.tolist())
            feat_vec.extend(boc2.tolist())
            feat_vec.extend([curr_tuple[2]])
            csv_file.write(",".join(str(feat) for feat in feat_vec) + "\n")


gen_features()

def gen_arff_file():
    with open("glove_embed_append.arff", 'w') as f:
        f.write("@RELATION Quora\n")
        for i in range(672):
            f.write("@ATTRIBUTE token_{} REAL\n".format(i))
        f.write("@ATTRIBUTE class {0,1}\n")
        cnt = 0
        f.write("\n@DATA\n")
        with open("../data/features_embed_append.csv", 'r', encoding="utf-8") as csv:
            for fv in csv:
                fv = fv.strip()
                features = fv.split(",")
                cnt += 1
                print(cnt)
                f.write(",".join(features) + "\n")
# gen_arff_file()


