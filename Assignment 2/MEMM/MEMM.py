import numpy as np
import nltk
from nltk.stem import PorterStemmer

def import_Train_Data():
	ftrain = open("./assignment2dataset/train.txt")
	lines = ftrain.readlines()

	Sentences = []
	Sentence_Construct = []
	for line in lines:
		if line =='\n':
			Sentences.append(Sentence_Construct)
			Sentence_Construct = []

		else:
			Sentence_Construct.append(line)

	ftrain.close()
	return Sentences


def import_Test_Data():
	ftest = open("./assignment2dataset/test.txt")
	lines = ftest.readlines()

	Sentences = []
	Sentence_Construct = []
	for line in lines:
		if line =='\n':
			Sentences.append(Sentence_Construct)
			Sentence_Construct = []

		else:
			Sentence_Construct.append(line)

	ftest.close()
	return Sentences


def Remove_Extra_Tag(Sentences):

	Final_Sentences = []

	for line in Sentences:

		##Adding start tag
		Final_Line = []

		for word in line:
			Split_Term = word.split(" ")

			if "-" in Split_Term[2]:
				Final_Word = Split_Term[0]+" "+Split_Term[1]+" "+Split_Term[2].split("-")[0]
				Final_Line.append(Final_Word)

			else:
				Final_Line.append(word[:-1])

		Final_Sentences.append(Final_Line)

	return Final_Sentences


def build_labelled_features(Sentences):

	####Feature Set being used
	# 3 POS tags -> PrevtopPrev Prev Current 
	# 2 chunk labels -> PrevtoPrev Prev
	# 3  word vectors -> PrevtoPrev Prev Current => Stems
	# Morphological Features -> isCaptital

	labelled_features = []
	
	for sentence in Sentences:

		for word_index in range(0,len(sentence)):

			split_sentence = sentence[word_index].split(" ")

			if word_index == 0:
				# print(sentence[word_index])
				cur_chunk = split_sentence[2]
				prev_to_prev_POS = 'st1'
				prev_POS = 'st2'
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = 'st1'
				prev_chunk = 'st2'
				cur_stem = PorterStemmer().stem(split_sentence[0])

			elif word_index == 1:
				# print(sentence[word_index])
				prev_split = sentence[word_index-1].split(" ")

				cur_chunk = split_sentence[2]
				prev_to_prev_POS = 'st2'
				prev_POS = prev_split[1]
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = 'st2'
				prev_chunk = prev_split[2]
				cur_stem = PorterStemmer().stem(split_sentence[0])
				
			else:
				prev_split1 = sentence[word_index-1].split(" ")
				prev_split2 = sentence[word_index-2].split(" ")

				cur_chunk = split_sentence[2]
				prev_to_prev_POS = prev_split2[1]
				prev_POS = prev_split2[1]
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = prev_split2[2]
				prev_chunk = prev_split1[2]
				cur_stem = PorterStemmer().stem(split_sentence[0])

			labelled_item = cur_chunk, prev_to_prev_POS, prev_POS, cur_POS, prev_to_prev_chunk, prev_chunk, cur_stem
			labelled_features.append(labelled_item)

	return labelled_features


def Generate_MEMM_features(labelled_feature):
	##can add capitalization

	features = {}
	
	features['prev_to_prev_POS'] = labelled_features[1]
	features['prev_POS'] = labelled_features[2]
	features['cur_POS'] = labelled_features[3]
	features['prev_to_prev_chunk'] = labelled_features[4]
	features['prev_chunk'] = labelled_features[5]
	features['cur_stem'] = labelled_features[6]

	return features


def train_maxent_classifier(labelled_features):


if __name__ == '__main__':

	Train_Sentences = import_Test_Data()
	Test_Sentences = import_Train_Data()

	Train_Sentences = Remove_Extra_Tag(Train_Sentences)
	Test_Sentences = Remove_Extra_Tag(Test_Sentences)

	labelled_features = build_labelled_features(Train_Sentences)

	print(labelled_features)

