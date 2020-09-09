import nltk
from nltk.corpus import brown
import random 
import numpy as np


def importdata():
	nltk.download('brown')
	nltk.download('universal_tagset')


def getRows():

	Sentences = nltk.corpus.brown.sents(categories=nltk.corpus.brown.categories())
	return Sentences


def getTaggedWords():

	return nltk.corpus.brown.tagged_words(tagset='universal')


def getTaggedSent():
	return nltk.corpus.brown.tagged_sents(categories= nltk.corpus.brown.categories(), tagset='universal')


def Parition_in_n(input_list, n):
    random.shuffle(input_list)
    return [input_list[i::n] for i in range(n)]



if __name__ == '__main__':

	importdata()

	List_of_Tags = []
	List_of_Words = []

	indextags = {}
	indexwords = {}

	Tagged_words = getTaggedWords()

	Tagged_Sentences = getTaggedSent()

	for words in Tagged_words:
		List_of_Tags.append(words[1])
		List_of_Words.append(words[0].lower())

	List_of_Words.append('st')
	List_of_Words.append('endt')

	List_of_Tags.append('st')
	List_of_Tags.append('endt')

	Set_of_Words = list(set(List_of_Words))
	Set_of_Tags = list(set(List_of_Tags))

	###########Smoothening Parameters###########
	Lambda = 0.01
	V = len(Set_of_Words)
	#############################################

	Dataset_Partition = Parition_in_n(list(Tagged_Sentences),5)

	#########Assigning Index to Words and Tagset for Probability Matrix######
	for x in range(0,len(Set_of_Tags)):
		indextags[Set_of_Tags[x]] = x

	for x in range(0,len(Set_of_Words)):
		indexwords[Set_of_Words[x]] = x


	#######################5 Cross Validation#########################
	Index_range = range(0,5)

	for test_index in Index_range:

		#####Probability Matrix##########
		prob_transition = [[0 for i in range(len(Set_of_Tags))] for j in range(len(Set_of_Tags))]
		prob_transition = np.asarray(prob_transition,dtype=np.float64)

		tagcount = [0 for i in range(len(Set_of_Tags))]
		tagcount = np.asarray(tagcount,dtype=np.float64)

		prob_emission = [[0 for i in range(len(Set_of_Words))] for j in range(len(Set_of_Tags))]
		prob_emission = np.asarray(prob_emission,dtype=np.float64)

		######Train the matrices
		Train_set = []

		for training_index in Index_range:

			if training_index == test_index:
				continue
			Train_set.extend(Dataset_Partition[training_index])

		
		for sentence in Train_set:

			sentence.insert(0,('st','st'))
			sentence.append(('endt','endt'))

			for word_index in range(0,len(sentence)-1):
				c1 = indextags[sentence[word_index][1]]
				c2 = indextags[sentence[word_index+1][1]]

				prob_transition[c1][c2]+=1


			for word_index in range(0,len(sentence)):
				c1 = indexwords[sentence[word_index][0].lower()]
				c2 = indextags[sentence[word_index][1]]

				prob_emission[c2][c1]+=1
				tagcount[c2]+=1


		for x in range(0,len(prob_transition)):
			if(np.sum(prob_transition[x])==0):
				continue
			prob_transition[x] = prob_transition[x]/np.sum(prob_transition[x])

		for x in range(0,len(tagcount)):
			if(tagcount[x]==0):
				continue
			prob_emission[x] = prob_emission[x]/tagcount[x]

		##################Smoothening######################
		prob_emission = prob_emission*(1-Lambda)
		prob_emission += (Lambda/V) 

		#################Testing Data######################
		Correct_Tags = 0.0
		Incorrect_Tags = 0.0

		for sentence in Dataset_Partition[test_index]:


			sentence.insert(0,('st','st'))
			sentence.append(('endt','endt'))

			word_sequence = [unit[0].lower() for unit in sentence]
			tag_sequence = [unit[1] for unit in sentence]

			Ans = [prob_emission.T[indexwords[word_sequence[0]]]]
			
			Ans_Tags = []

			for x in range(1,len(word_sequence)):

				Best_prob = [0 for i in range(len(Set_of_Tags))]
				Best_prob = np.asarray(Best_prob,dtype=np.float64)

				Backtrack_Tag = [0 for i in range(len(Set_of_Tags))]
				Backtrack_Tag = np.asarray(Backtrack_Tag,dtype=int)

				current_emission = prob_emission.T[indexwords[word_sequence[x]]]


				for tag1 in range(0,len(Set_of_Tags)):
					for tag2 in range(0,len(Set_of_Tags)):

						if(Best_prob[tag1]<(Ans[-1][tag2]*prob_transition[tag2][tag1]*current_emission[tag1])):
							Best_prob[tag1] = Ans[-1][tag2]*prob_transition[tag2][tag1]*current_emission[tag1]
							Backtrack_Tag[tag1] = tag2



				Ans.append(np.array(Best_prob))
				Ans_Tags.append(np.array(Backtrack_Tag))


			
			final_tags = []

			max_arg = Ans[-1].argmax()

			final_tag_index = []
			final_tag_index.append(max_arg)

			for x in range(2,len(word_sequence)):
				max_arg = Ans_Tags[len(Ans)-x][max_arg] 
				final_tag_index.append(max_arg)

			for x in range(len(final_tag_index)-1,0,-1):
				final_tags.append(Set_of_Tags[final_tag_index[x]])

			tag_sequence = tag_sequence[1:-1]
			
			###########Calculating Accuracy#################
			for t in range(0,len(final_tags)):

				if final_tags[t] == tag_sequence[t]:
					Correct_Tags = Correct_Tags+1

				else:
					Incorrect_Tags = Incorrect_Tags+1
			#################################################

		Accuracy = Correct_Tags/(Correct_Tags+Incorrect_Tags)

		print(Accuracy)



		




	
