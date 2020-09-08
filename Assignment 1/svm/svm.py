import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import brown
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import random
import numpy as np
import time

import seaborn as sns
sns.set()

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class Observations:
	def __init__(self):
		self.accuracy = 0.0
		self.per_pos_acc = dict()
		self.conf_matx = np.zeros((2, 2))


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

def getSent():
	return nltk.corpus.brown.sents()

def get_word_embedding_model():
	model = gensim.models.Word2Vec(getSent(), size=100, window=5, min_count=1, workers=4)
	return model

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

def getFeatureData(sentence_dataset, word_vect, vectorizer = DictVectorizer(), refit = True):
	stemmer = SnowballStemmer("english", ignore_stopwords=True)
	feature_vec = []
	word_embeds = []
	pos_tags = []

	for sent in sentence_dataset:
		for i in range(len(sent)):
			features = {
			# 'prev-word' : '' if i==0 else sent[i-1][0],
			'prev-tag' : '' if i==0 else sent[i-1][1],
			# 'word' : word_vect[sent[i][0]],
			# 'tag' : sent[i][1],
			# 'next-word' : '' if i==len(sent)-1 else sent[i+1][0],
			'next-tag' : '' if i==len(sent)-1 else sent[i+1][1],
			'is_first' : i==0,
			'is_last' : i==len(sent)-1,
			'is_capitalized' : sent[i][0][0].upper == sent[i][0][0],
			'is_numeric' : sent[i][0].isdigit(),
			# 'prefix-1' : sent[i][0][0],
			'prefix-2' : '' if len(sent[i][0]) < 2  else sent[i][0][:1],
			# 'suffix-1' : sent[i][0][-1],
			'suffix-2' : '' if len(sent[i][0]) < 2  else sent[i][0][-2:]
			}
			word_embeds.append(word_vect[sent[i][0]])
			feature_vec.append(features)
			pos_tags.append(sent[i][1])
	print("Done")

	if refit:
		vectorizer.fit(feature_vec)
	
	vectorized_features = vectorizer.transform(feature_vec).toarray()
	all_features = [np.array(list(vectorized_features[i])+list(word_embeds[i])) for i in range(len(word_embeds))]
	print(len(all_features[0]))
	return vectorizer, np.array(all_features), np.array(pos_tags)

def train(num_cross_valid, word_vect):
	dataset, words, tags = getData(num_cross_valid)
	print(len(tags))

	for k in range(num_cross_valid):
		print(len(dataset[k]))

	observations = []

	for i in range(1):
		obs = Observations()
		train_set = []
		test_set = []
		for j in range(num_cross_valid):
			if i!=j:
				train_set = train_set + dataset[j]
			else:
				test_set = test_set + dataset[j]
		vectorizer, feature_vecs, pos_tags = getFeatureData(train_set[0:1000], word_vect)
		_, test_vecs, test_pos = getFeatureData(test_set[0:500], word_vect, vectorizer, False)
		
		print("Training started len feature", len(feature_vecs), len(test_vecs))
		

		# linear = make_pipeline(StandardScaler(), LinearSVC(tol=1e-3, random_state=0, max_iter=50, verbose=1, dual=False))
		

		# linear.fit(feature_vecs, pos_tags)
		# O vs R

		tag_classifiers = []
		ws = []
		bs = []
		tags2 = []
		for cnt, k in enumerate(tags):
			tg_d = []
			vl = 0
			for l in pos_tags:
				if l==k:
					vl += 1
					tg_d.append(-1)
				else:
					tg_d.append(1)
			if vl == 0:
				print("nf", k)
				continue
			tags2.append(k)
			print("Tag ", k, cnt, " training started. Num tags", vl)
			# ws.append(svm_train(feature_vecs, tg_d))
			w, b = cvxopt_train(feature_vecs, np.array(tg_d))
			ws.append(w)
			bs.append(b)
			print("Internediate training for", cnt)
			svm_cl = SVC(kernel='linear', C=1, verbose=1, probability=True).fit(feature_vecs, tg_d)
			print("Tag ", k, " training finished")
			tag_classifiers.append(svm_cl)

		print("Part training finished")
		# linear = SVC(kernel='linear', C=1, decision_function_shape='ovr', verbose=1).fit(feature_vecs, pos_tags)
		print("Training ended")

		print("Test started")
		outs = []
		outs1 = []
		for cnt, _ in enumerate(tags2):
			print("Part Test started for ", cnt)
			# outs1.append(predict(ws[cnt], test_vecs))
			outs1.append(cvxopt_predict(ws[cnt], bs[cnt], test_vecs))
			print("Intermediate test for", cnt)
			outs.append(tag_classifiers[cnt].predict_proba(test_vecs))
		print("Part Test finished")

		outs = np.array(outs)
		outs1 = np.array(outs1)
		print("results")
		print(outs[0], outs1[0], test_pos)

		outs_tmp = []
		for kk in outs:
			c = []
			for i in range(len(kk)):
				if kk[i][0]>kk[i][1]:
					c.append(np.array([1, 0]))
				else:
					c.append(np.array([0, 1]))
			outs_tmp.append(np.array(c))

		outs = np.array(outs_tmp)

		out_tags = []
		out_tags1 = []

		for cnt, k in enumerate(test_vecs):
			# print("Tagging started for ", cnt)
			out_tags1.append(tags2[np.argmax(outs1[:, cnt, 0])])
			out_tags.append(tags2[np.argmax(outs[:, cnt, 0])])
		print("Output started")


		# linear_pred = linear.predict(test_vecs)
		# print(sum(linear_pred == test_pos), len(test_pos))
		print(sum(out_tags1 == test_pos), len(test_pos))
		print(sum(out_tags == test_pos), len(test_pos))
		accuracy_lin = linear.score(test_vecs, test_pos)
		print('Accuracy Linear Kernel:', accuracy_lin)



		# print(len(feature_vecs.toarray()[0]), pos_tags[0])

		# Train using svm sklearn


		#Test accuracy

def cvxopt_train(X, y):
	C = 2
	m,n = X.shape
	y = y.reshape(-1,1) * 1.
	X_dash = y * X
	H = np.dot(X_dash , X_dash.T) * 1.

	#Converting into cvxopt format - as previously
	P = cvxopt_matrix(H)
	q = cvxopt_matrix(-np.ones((m, 1)))
	G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
	h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
	A = cvxopt_matrix(y.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))

	#Run solver
	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])

	#==================Computing and printing parameters===============================#
	w = ((y * alphas).T @ X).reshape(-1,1)
	S = (alphas > 1e-4).flatten()
	b = y[S] - np.dot(X[S], w)
	return w, b


def svm_train(x, y):
	#add bias to sample vectors
	x = np.c_[x,np.ones(len(x))]

	#initialize weight vector
	w = np.zeros(len(x[0]))
	#learning rate 
	lam = 0.001
	#array of number for shuffling
	order = np.arange(0,len(x),1)
	print(len(order))

	margin_current = 0
	margin_previous = -10

	pos_support_vectors = 0
	neg_support_vectors = 0

	not_converged = True
	t =0 
	start_time = time.time()

	while(not_converged):
		margin_previous = margin_current
		t += 1
		pos_support_vectors = 0
		neg_support_vectors = 0
		
		eta = 1/(lam*t)
		fac = (1-(eta*lam))*w
		random.shuffle(order)
		for i in order:
			# print(i, " i")
			prediction = np.dot(x[i],w)
			
			#check for support vectors
			if (round((prediction),1) == 1):
				pos_support_vectors += 1
				#pos support vec found
			if (round((prediction),1) == -1):
				neg_support_vectors += 1
				#neg support vec found
				
			#misclassification
			if (y[i]*prediction) < 1 :
				w = fac + eta*y[i]*x[i]            
			#correct classification
			else:
				w = fac
		if(t%200==0):
			print(t, " t")
		if(t>6000):
			break
		if(t>4000):
			margin_current = np.linalg.norm(w)
			# not_converged = False    
			if((pos_support_vectors > 0) and (neg_support_vectors > 0) and ((margin_current - margin_previous) < 0.01)):
				not_converged = False
	#print running time
	print("--- %s seconds ---" % (time.time() - start_time))
	return w
	pass


def cvxopt_predict(w, b, x_test):
	y_pred = []
	# x_test = np.c_[x_test,np.ones(len(x_test))]
	for i in x_test:
		pred = np.dot(w.T,i) + b
		if(pred[0][0] > 0):
			y_pred.append(np.array([0, 1]))
		elif(pred[0][0] < 0):
			y_pred.append(np.array([1, 0]))
	return y_pred

def predict(w, x_test):
	y_pred = []
	x_test = np.c_[x_test,np.ones(len(x_test))]
	for i in x_test:
		pred = np.dot(w,i)
		if(pred > 0):
			y_pred.append(np.array([0, 1]))
		elif(pred < 0):
			y_pred.append(np.array([1, 0]))
	return y_pred


if __name__ == '__main__':
	importdata()

	# Run once
	# model = get_word_embedding_model()
	# word_vect = model.wv
	# print(word_vect['the'])
	
	fname = get_tmpfile("word_vector.kv")
	# word_vect.save(fname)
	word_vect = KeyedVectors.load(fname, mmap='r')
	# print(word_vect['the'])
	train(5, word_vect)

	

