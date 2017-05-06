import pickle
import sys
import re
import spell
from spell import correction
import csv
import numpy as np
from string import ascii_lowercase as ascii_l
import random
from sklearn.svm import LinearSVC, SVC

def load_quora_data(infile):
    cnt = 0
    rows = []
    with open(infile, 'r', encoding='utf-8') as input:
        input = csv.reader(input, delimiter=',')
        for idx, row in enumerate(input):
            if idx > 0:  # First row is fields data
                rows.append(row)
        # for line in input:
        #     if cnt == 0 :
        #         cnt += 1
        #         continue
        #     else:
        #         rows.append(line.split(","))

    return rows

# load_quora_data("../data/QuoraData.csv")

def saving_glove_words_as_dict(infile):
    dict = {}
    cnt = 0
    with open(infile, 'r', encoding='utf-8') as input:
        for line in input:
            cnt += 1
            print(cnt)
            word = line.split(" ")[0]
            dict[word] = 1

    with open("../data/glove_dict.p", "wb") as f:
        pickle.dump(dict, f)

# saving_glove_words_as_dict("../../local_copy/data/glove.42B.300d/glove.42B.300d.txt")
def process_sentence(input_text):
    input_text = input_text.strip(" ?")
    input_text = input_text.replace("/", " ")
    input_text = input_text.replace("-", " ")
    input_text = input_text.replace("(", " ")
    input_text = input_text.replace(")", " ")
    input_text = input_text.replace(".", " ")
    input_text = input_text.replace("?", " ")
    input_text = input_text.replace(",", " ")
    input_text = input_text.replace(";", " ")
    input_text.strip()
    input_text = re.sub(r'[^a-zA-Z0-9 ]', '', input_text)
    return input_text

def process_data():
    total_num_tokens = 0
    all_words_dict = {}
    recog_cnt = 0
    data_tuples = []
    rows = load_quora_data("../data/QuoraData.csv")
    with open("../data/glove_dict.p", "rb") as f:
        glove_dict = pickle.load(f)

    for idx, row in enumerate(rows):
        if(idx + 1)%1000 == 0:
            print("Abount to process example", (idx+1))
            sys.stdout.flush()
        sent1 = process_sentence(row[3])
        words1 = sent1.split()
        final_words1 = []

        sent2 = process_sentence(row[4])
        words2 = sent2.split()
        final_words2 = []

        for word in words1:
            total_num_tokens += 1
            word = word.lower()
            if word not in all_words_dict:
                if word in glove_dict:
                    recog_cnt += 1
                    final_words1.append(word)
                    continue
                correction_word = correction(word)
                if correction_word in glove_dict:
                    final_words1.append(correction_word)
                    if correction_word not in all_words_dict:
                        all_words_dict[correction_word] = 1
                        recog_cnt += 1
                else:
                    all_words_dict[word] = 1
                    final_words1.append(word)
            else:
                final_words1.append(word)

        for word in words2:
            total_num_tokens += 1
            word = word.lower()
            if word not in all_words_dict:
                if word in glove_dict:
                    recog_cnt += 1
                    final_words2.append(word)
                    continue
                correction_word = correction(word)
                if correction_word in glove_dict:
                    final_words2.append(correction_word)
                    if correction_word not in all_words_dict:
                        all_words_dict[correction_word] = 1
                        recog_cnt += 1
                else:
                    all_words_dict[word] = 1
                    final_words2.append(word)
            else:
                final_words2.append(word)

        processed_sentence_1 = " ".join(final_words1)
        processed_sentence_2 = " ".join(final_words2)

        curr_label = int(row[5].strip())
        data_tuples.append((processed_sentence_1, processed_sentence_2, curr_label))
    print("number of data tuples collected", len(data_tuples))

    with open("../data/data_tuples_gloveem.p",'wb') as f:
        pickle.dump(data_tuples,f)

    print("Total number of words in the data : ", len(all_words_dict))
    print("Number of words in the data that are in glove", recog_cnt)


# process_data()

def gen_embeddings():
    with open("../data/data_tuples_gloveem.p", "rb") as f:
        data_tuples = pickle.load(f)

    needed_glove_words = {}
    for tup in data_tuples:
        s1 = tup[0]
        s2 = tup[1]
        for word in s1.split():
            if word not in needed_glove_words:
                needed_glove_words[word] = 1
        for word in s2.split():
            if word not in needed_glove_words:
                needed_glove_words[word] = 1

    print("Num of needed glove words : ", len(needed_glove_words))
    needed_glove_dict = {}
    num_words = 0
    vec_file = open("../data/glove.42B.300d.txt", encoding="utf-8")
    for line in vec_file:
        words = line.split()
        if words[0] in needed_glove_words:
            needed_glove_dict[words[0]] = np.asarray(words[1:])

        num_words += 1
        if num_words % 100000 == 0:
            print("Processed so far: ", num_words / 100000, " x100K")

    vec_file.close()

    print("Done creating glove dict for necessary words")
    print("number of words", len(needed_glove_dict))

    with open("../data/needed_glove_dict.p", "wb") as f:
        pickle.dump(needed_glove_dict,f)
# gen_embeddings()

def gen_features():
    with open("../data/data_tuples_gloveem.p", "rb") as f:
        data_tuples = pickle.load(f)

    with open("../data/needed_glove_dict.p", "rb") as f:
        glove_dict = pickle.load(f)
    all_chars = []
    for ch in ascii_l:
        all_chars.append(ch)

    for idx in range(10):
        all_chars.append(str(idx))

    char_to_idx = {ch: ix for ix, ch in enumerate(all_chars)}
    num_examples = len(data_tuples)
    glove_dim = 300
    boc_dim = 36
    curr_idx = 0
    feat_dim = glove_dim * 2 + boc_dim * 2  # GLOVE DIMENSION + BOC DIM
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
            print
            "Done converting tuples to feat matrix"
            print
            "Feat matrix size ", feat_matrix.shape
            print
            "Last idx stored ", curr_idx - 1

            # Make the train/test split 80-20
            train_pc = 0.8
            test_pc = 1 - train_pc

            idxes = np.arange(num_examples)
            random.seed(50)
            random.shuffle(idxes)

            # Training data
            train_matrix = feat_matrix[idxes[0: int(np.ceil(train_pc * num_examples))]]
            train_labels = labels[idxes[0: int(np.ceil(train_pc * num_examples))]]

            # Val data
            val_matrix = feat_matrix[idxes[int(np.ceil(train_pc * num_examples)):]]
            val_labels = labels[idxes[int(np.ceil(train_pc * num_examples)):]]

            clf = LinearSVC(C=10.0, verbose=1)
            # clf = SVC(C=10.0, verbose=1, kernel='poly', degree=2)
            print
            "About to train SVM"
            clf.fit(train_matrix, train_labels)
            print
            "Done training"

            val_pred = clf.predict(val_matrix)
            acc = np.mean(val_labels == val_pred)

            print
            "Validation Accuracy is ", acc
            print
            "Training Accuracy is ", np.mean(train_labels == clf.predict(train_matrix))

            # feat_vec = []
            # feat_vec.extend(vec1.tolist())
            # feat_vec.extend(vec2.tolist())
            # feat_vec.extend(boc1.tolist())
            # feat_vec.extend(boc2.tolist())
            # feat_vec.extend([curr_tuple[2]])
            # csv_file.write(",".join(str(feat) for feat in feat_vec) + "\n")


gen_features()

def gen_arff_file():
    with open("glove_embed_append.arff", 'w') as f:
        f.write("@RELATION Quora\n")
        for i in range(672):
            f.write("@ATTRIBUTE token_{} REAL\n".format(i))
        f.write("@ATTRIBUTE class {0,1}\n")
        cnt = 0
        # Data instances
        f.write("\n@DATA\n")
        with open("../data/features_embed_append.csv", 'r', encoding="utf-8") as csv:
            for fv in csv:
                fv = fv.strip()
                features = fv.split(",")
                cnt += 1
                print(cnt)
                f.write(",".join(features) + "\n")
# gen_arff_file()