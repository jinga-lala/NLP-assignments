import nltk
from nltk.corpus import brown
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

def importdata():
	nltk.download('brown')
	nltk.download('stopwords')
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

def getData(num_cross_valid):
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

	Dataset_Partition = Parition_in_n(list(Tagged_Sentences),num_cross_valid)

	return Dataset_Partition, Set_of_Words, Set_of_Tags

	# print(Set_of_Words[0:100], Set_of_Tags)
	# print(Dataset_Partition[0][0])

def getFeatureData(sentence_dataset, vectorizer = DictVectorizer(), refit = True):
	stemmer = SnowballStemmer("english", ignore_stopwords=True)
	feature_vec = []
	pos_tags = []

	for sent in sentence_dataset:
		for i in range(len(sent)):
			features = {
			# 'prev-word' : '' if i==0 else sent[i-1][0],
			'prev-tag' : '' if i==0 else sent[i-1][1],
			# 'word' : stemmer.stem(sent[i][0]),
			# 'tag' : sent[i][1],
			# 'next-word' : '' if i==len(sent)-1 else sent[i+1][0],
			'next-tag' : '' if i==len(sent)-1 else sent[i+1][1],
			'is_first' : i==0,
			'is_last' : i==len(sent)-1,
			'is_capitalized' : sent[i][0][0].upper == sent[i][0][0],
			'is_numeric' : sent[i][0].isdigit(),
			'prefix-1' : sent[i][0][0],
	        'prefix-2' : '' if len(sent[i][0]) < 2  else sent[i][0][:1],
	        'suffix-1' : sent[i][0][-1],
	        'suffix-2' : '' if len(sent[i][0]) < 2  else sent[i][0][-2:]
			}

			feature_vec.append(features)
			pos_tags.append(sent[i][1])
	print("Done")

	if refit:
		vectorizer.fit(feature_vec)
	
	vectorized_features = vectorizer.transform(feature_vec)
	print(len(vectorized_features.toarray()[0]))
	return vectorizer, vectorized_features, pos_tags

def train(num_cross_valid):
	dataset, words, tags = getData(num_cross_valid)

	for k in range(num_cross_valid):
		print(len(dataset[k]))

	for i in range(1):
		train_set = []
		test_set = []
		for j in range(num_cross_valid):
			if i!=j:
				train_set = train_set + dataset[j]
			else:
				test_set = test_set + dataset[j]

		vectorizer, feature_vecs, pos_tags = getFeatureData(train_set[0:10000])
		
		print("Training started")
		linear = make_pipeline(StandardScaler(), LinearSVC(tol=1e-3, random_state=0, max_iter=50, verbose=1, dual=False))
		linear.fit(feature_vecs.toarray(), np.array(pos_tags))
		# linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr', verbose=1).fit(feature_vecs, pos_tags)
		print("Training ended")

		_, test_vecs, test_pos = getFeatureData(test_set[0:1000], vectorizer, False)

		linear_pred = linear.predict(test_vecs.toarray())
		print(sum(linear_pred == test_pos), len(test_pos))
		accuracy_lin = linear.score(test_vecs.toarray(), test_pos)
		print('Accuracy Linear Kernel:', accuracy_lin)

		# print(len(feature_vecs.toarray()[0]), pos_tags[0])

		# Train using svm sklearn


		#Test accuracy


if __name__ == '__main__':
	importdata()
	train(5)
	

