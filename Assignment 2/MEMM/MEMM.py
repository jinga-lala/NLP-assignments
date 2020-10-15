import numpy as np
from numpy import unravel_index
import nltk
from nltk.stem import PorterStemmer
from nltk.classify import MaxentClassifier
import pickle

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

def Get_Tagset(Sentences):

	TagSet = []

	for sentence in Sentences:
		for word in sentence:
			TagSet.append(word.split(" ")[2])

	return set(TagSet)


def build_labelled_features(Sentences):

	####Feature Set being used
	# 3 POS tags -> PrevtopPrev Prev Current 
	# 2 chunk labels -> PrevtoPrev Prev
	# 3  word-stems -> PrevtoPrev Prev Current => Stems
	# Morphological Features -> isCaptital

	labelled_features = []
	
	for sentence in Sentences:

		for word_index in range(0,len(sentence)):

			split_sentence = sentence[word_index].split(" ")

			if word_index == 0:
				# print(sentence[word_index])
				cur_chunk = split_sentence[2]
				prev_to_prev_POS = 'st'
				prev_POS = 'st'
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = 'st'
				prev_chunk = 'st'
				cur_stem = PorterStemmer().stem(split_sentence[0].lower())
				prev_stem = 'st'
				prev_to_prev_stem = 'st'

			elif word_index == 1:
				# print(sentence[word_index])
				prev_split = sentence[word_index-1].split(" ")

				cur_chunk = split_sentence[2]
				prev_to_prev_POS = 'st'
				prev_POS = prev_split[1]
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = 'st'
				prev_chunk = prev_split[2]
				cur_stem = PorterStemmer().stem(split_sentence[0].lower())
				prev_stem = PorterStemmer().stem(prev_split[0].lower())
				prev_to_prev_stem = 'st'

				
			else:
				prev_split1 = sentence[word_index-1].split(" ")
				prev_split2 = sentence[word_index-2].split(" ")

				cur_chunk = split_sentence[2]
				prev_to_prev_POS = prev_split2[1]
				prev_POS = prev_split1[1]
				cur_POS = split_sentence[1]
				prev_to_prev_chunk = prev_split2[2]
				prev_chunk = prev_split1[2]
				cur_stem = PorterStemmer().stem(split_sentence[0].lower())
				prev_stem = PorterStemmer().stem(prev_split1[0].lower())
				prev_to_prev_stem = PorterStemmer().stem(prev_split2[0])

			labelled_item = cur_chunk, prev_to_prev_POS, prev_POS, cur_POS, prev_to_prev_chunk, prev_chunk, cur_stem, prev_stem, prev_to_prev_stem
			labelled_features.append(labelled_item)

	return labelled_features


def Generate_MEMM_features(input_feature):
	##can add capitalization

	features = {}
	
	features['prev_to_prev_POS'] = input_feature[1]
	features['prev_POS'] = input_feature[2]
	features['cur_POS'] = input_feature[3]
	features['prev_to_prev_chunk'] = input_feature[4]
	features['prev_chunk'] = input_feature[5]
	features['cur_stem'] = input_feature[6]
	features['prev_stem'] = input_feature[7]
	features['prev_to_prev_stem'] = input_feature[8]

	return features


def train_maxent_classifier(labelled_features):

	train_set = []
	for lf in labelled_features:

		train_set.append((Generate_MEMM_features(lf), lf[0]))

	maxent_classifier = MaxentClassifier.train(train_set, max_iter=10)
	return maxent_classifier


if __name__ == '__main__':

	Train_Sentences = import_Test_Data()
	Test_Sentences = import_Train_Data()

	Train_Sentences = Remove_Extra_Tag(Train_Sentences)
	Test_Sentences = Remove_Extra_Tag(Test_Sentences)

	Set_of_Tags = list(Get_Tagset(Train_Sentences))
	Set_of_Tags.append('st')
	Tag_Dict = {}

	for tag_ind in range(0,len(Set_of_Tags)):
		Tag_Dict[Set_of_Tags[tag_ind]] = tag_ind

	labelled_features = build_labelled_features(Train_Sentences)

	# maxent_classifier = train_maxent_classifier(labelled_features)

	f = open("my_classifier.pickle", "rb")
	maxent_classifier = pickle.load(f)

	#####Confusion Matrix
	confusion_matrix = [[0 for i in range(len(Set_of_Tags))] for j in range(len(Set_of_Tags))]
	confusion_matrix = np.asarray(confusion_matrix,dtype=np.float64)

	#####Viterbi Algorithm
	i = 0

	Correct_Tags = 0.0
	Incorrect_Tags = 0.0

	for sentence in Test_Sentences:

		sentence.insert(0,'st st st')
		sentence.insert(0,'st st st')

		stem_sequence = [PorterStemmer().stem(unit.split(" ")[0].lower()) for unit in sentence]
		POStag_sequence = [unit.split(" ")[1] for unit in sentence]
		Chunktag_sequence = [unit.split(" ")[2] for unit in sentence]

		###Since there are 4 possibilities BB, BI, IB, II hence we will use these 4

		Ans = [np.asarray([[0 for tag_prev in Set_of_Tags] for tag_prev_to_prev in Set_of_Tags],dtype=np.float64)]
		Ans[0][Tag_Dict['B']][Tag_Dict['st']] += 1
		
		Ans_Tags = []

		for x in range(3,len(stem_sequence)):

			Best_prob = [[0 for i2 in range(len(Set_of_Tags))] for i1 in range(len(Set_of_Tags))]
			Best_prob = np.asarray(Best_prob,dtype=np.float64)

			Backtrack_Tag = [[0 for i2 in range(len(Set_of_Tags))] for i1 in range(len(Set_of_Tags))]
			Backtrack_Tag = np.asarray(Backtrack_Tag,dtype=int)


			for tag_prev in range(0,len(Set_of_Tags)):
				for tag_prev_to_prev in range(0,len(Set_of_Tags)):

					if Ans[-1][tag_prev][tag_prev_to_prev] == 0:
						continue

					###cur_chunk, prev_to_prev_POS, prev_POS, cur_POS, prev_to_prev_chunk, prev_chunk, cur_stem, prev_stem, prev_to_prev_stem
					input_feature = Chunktag_sequence[x], POStag_sequence[x-2], POStag_sequence[x-1], POStag_sequence[x], Set_of_Tags[tag_prev_to_prev], Set_of_Tags[tag_prev], stem_sequence[x], stem_sequence[x-1], stem_sequence[x-2]

					Maxent_probablity = maxent_classifier.prob_classify(Generate_MEMM_features(input_feature))

					for tag_ind in range(0,len(Set_of_Tags)):
						if Best_prob[tag_ind][tag_prev] < Ans[-1][tag_prev][tag_prev_to_prev]*Maxent_probablity.prob(Set_of_Tags[tag_ind]):
							Best_prob[tag_ind][tag_prev] = Ans[-1][tag_prev][tag_prev_to_prev]*Maxent_probablity.prob(Set_of_Tags[tag_ind])
							Backtrack_Tag[tag_ind][tag_prev] = tag_prev_to_prev

					# if(Best_prob[tag1]<(Ans[-1][tag2]*prob_transition[tag2][tag1]*current_emission[tag1])):
					# 	Best_prob[tag1] = Ans[-1][tag2]*prob_transition[tag2][tag1]*current_emission[tag1]
					# 	Backtrack_Tag[tag1] = tag2


			Ans.append(np.array(Best_prob))
			Ans_Tags.append(np.array(Backtrack_Tag))


		
		final_tags = []

		array_shape = Ans[-1].shape

		max_arg = Ans[-1].argmax()
		max_arg = unravel_index(max_arg, array_shape)

		cur_chunk_tag = max_arg[0]
		prev_chunk_tag = max_arg[1]

		final_tag_index = []
		final_tag_index.append(cur_chunk_tag)
		final_tag_index.append(prev_chunk_tag)

		for x in range(2,len(stem_sequence)-1):

			prev_to_prev_chunk_tag = Ans_Tags[len(Ans)-x][cur_chunk_tag][prev_chunk_tag] 
			final_tag_index.append(prev_to_prev_chunk_tag)
			cur_chunk_tag = prev_chunk_tag
			prev_chunk_tag = prev_to_prev_chunk_tag

		for x in range(len(final_tag_index)-1,-1,-1):
			final_tags.append(Set_of_Tags[final_tag_index[x]])

		###########Calculating Accuracy#################
		for t in range(1,len(final_tags)):

			####Updating Confusion matrix
			confusion_matrix[Tag_Dict[Chunktag_sequence[t]]][Tag_Dict[final_tags[t]]] = confusion_matrix[Tag_Dict[Chunktag_sequence[t]]][Tag_Dict[final_tags[t]]]+1

			if final_tags[t] == Chunktag_sequence[t+1]:
				Correct_Tags = Correct_Tags+1

			else:
				Incorrect_Tags = Incorrect_Tags+1
			
		#################################################

	Accuracy = Correct_Tags/(Correct_Tags+Incorrect_Tags)

	print(Accuracy)


	

