import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import brown
# from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC               # Importing this only for comparison purposes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import random
import numpy as np
import time

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

import seaborn as sn
import matplotlib.pyplot as plt 

class Observations:
	def __init__(self):
		self.train_accuracy = 0.0
		self.accuracy = 0.0
		self.per_pos_acc = dict()
		self.conf_matx = dict()
		self.conf = None
		self.labels = None

def importdata():
	nltk.download('brown')
	# nltk.download('stopwords')
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

	List_of_Tags.append('st')
	List_of_Tags.append('endt')

	Set_of_Words = list(set(List_of_Words))
	Set_of_Tags = list(set(List_of_Tags))

	Dataset_Partition = Parition_in_n(list(Tagged_Sentences),num_cross_valid)

	return Dataset_Partition, Set_of_Words, Set_of_Tags

def getFeatureData(sentence_dataset, word_vect, vectorizer = DictVectorizer(), refit = True):
	# stemmer = SnowballStemmer("english", ignore_stopwords=True)
	feature_vec = []
	word_embeds = []
	pos_tags = []

	for sent in sentence_dataset:
		for i in range(len(sent)):
			features = {
			'prev-tag' : '' if i==0 else sent[i-1][1],
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
			# curr_wrd_embed = list(word_vect[sent[i][0]])
			# if i==0:
			# 	curr_wrd_embed = curr_wrd_embed + list(word_vect["."])
			# else:
			# 	curr_wrd_embed = curr_wrd_embed + list(word_vect[sent[i-1][0]])
			word_embeds.append(word_vect[sent[i][0]])
			feature_vec.append(features)
			pos_tags.append(sent[i][1])

	if refit:
		vectorizer.fit(feature_vec)
	
	vectorized_features = vectorizer.transform(feature_vec).toarray()
	all_features = [np.array(list(vectorized_features[i])+list(word_embeds[i])) for i in range(len(word_embeds))]
	return vectorizer, np.array(all_features), np.array(pos_tags)

def svm(num_cross_valid, word_vect):
	print("Fetch Data...")
	dataset, words, tags = getData(num_cross_valid)
	print("Dataset contains ", len(words), "distinct words and", len(tags), " distinct tags")

	observations = []

	print("Begin ", num_cross_valid, "-fold cross validation training")
	for i in range(num_cross_valid):
		phase = i
		obs = Observations()
		train_set = []
		test_set = []
		for j in range(num_cross_valid):
			if i!=j:
				train_set = train_set + dataset[j]
			else:
				test_set = test_set + dataset[j]

		vectorizer, feature_vecs, pos_tags = getFeatureData(train_set[:120], word_vect)
		mean = np.mean(feature_vecs)
		scale = np.std(feature_vecs)
		feature_vecs = (feature_vecs - mean) / scale
		print("Using only", 100, "sentences for training svm model")
		print("Training Dataset size:", 100)
		print("Number of word (feature) vectors:", len(feature_vecs))
		print("Feature length: ", len(feature_vecs[0]))

		_, test_vecs, test_pos = getFeatureData(test_set[0:1000], word_vect, vectorizer, False)
		# mean = np.mean(test_vecs)
		# scale = np.std(test_vecs)
		test_vecs = (test_vecs - mean) / scale
		print("Using ", 1000, "sentences for testing svm model")
		print("Test Dataset size:", 1000)
		print("Number of word (feature) vectors:", len(test_vecs))
		print("Feature length: ", len(test_vecs[0]))

		print("Begin training one vs rest classifiers for each tag...")
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
				print("No feature vector for tag-", k, "found thus skipping training for this tag.")
				continue
			tags2.append(k)
			print("Num feature (word) vectors observed for tag-", k, "is", vl)
			print("Training for tag-", k, " started.")
			w, b = cvxopt_train(feature_vecs, np.array(tg_d))
			ws.append(w)
			bs.append(b)

			print("\n\n Training for sklearn's svc...")
			svm_cl = SVC(kernel='linear', C=1, verbose=1, probability=True).fit(feature_vecs, tg_d)
			tag_classifiers.append(svm_cl)
			print("Training for tag ", k, " finished.")
			print(len(tags)-cnt-1, "tags left for training.")

		print("\n\nIndividual tag classifiers training finished.")
		# linear = SVC(kernel='linear', C=1, decision_function_shape='ovr', verbose=1).fit(feature_vecs, pos_tags)
		print("Training ends.")

		print("\nTesting started...")
		outs = []
		outs1 = []
		train_outs = []
		train_outs1 = []
		for cnt, tgs in enumerate(tags2):
			print("Predicting tags for test data...")
			print("Predicting probability for tag-", tgs)
			train_outs1.append(cvxopt_predict(ws[cnt], bs[cnt], feature_vecs))
			outs1.append(cvxopt_predict(ws[cnt], bs[cnt], test_vecs))
			print("Predicting probability for sklearn's svc")
			train_outs.append(tag_classifiers[cnt].predict_proba(feature_vecs))
			outs.append(tag_classifiers[cnt].predict_proba(test_vecs))
			print(len(tags2)-cnt-1, "left to test.")
		print("Prediction finished.")

		outs = np.array(outs)
		outs1 = np.array(outs1)
		train_outs = np.array(train_outs)
		train_outs1 = np.array(train_outs1)

		# outs_tmp = []
		# for kk in outs:
		# 	c = []
		# 	for i in range(len(kk)):
		# 		if kk[i][0]>kk[i][1]:
		# 			c.append(np.array([1, 0]))
		# 		else:
		# 			c.append(np.array([0, 1]))
		# 	outs_tmp.append(np.array(c))

		# outs = np.array(outs_tmp)

		out_tags = []
		out_tags1 = []
		train_out_tags = []
		train_out_tags1 = []

		for cnt, k in enumerate(test_vecs):
			out_tags1.append(tags2[np.argmax(outs1[:, cnt, 0])])
			out_tags.append(tags2[np.argmax(outs[:, cnt, 0])])

		for cnt, k in enumerate(feature_vecs):
			train_out_tags1.append(tags2[np.argmax(train_outs1[:, cnt, 0])])
			train_out_tags.append(tags2[np.argmax(train_outs[:, cnt, 0])])

		print("Results for phase", phase+1, " out of 5.")

		# linear_pred = linear.predict(test_vecs)
		# print(sum(linear_pred == test_pos), len(test_pos))
		print(sum(out_tags1 == test_pos), len(test_pos))
		print(sum(out_tags == test_pos), len(test_pos))
		
		# record observation
		obs.train_accuracy = (sum(train_out_tags1 == pos_tags) * 100) / len(pos_tags)
		obs.accuracy = (sum(out_tags1 == test_pos) * 100) / len(test_pos)
		print("Sklearn's svc training accuracy :", (sum(train_out_tags == pos_tags)*100)/len(pos_tags))
		print("Our training accuracy :", obs.train_accuracy)
		print("Sklearn's svc accuracy :", (sum(out_tags == test_pos)*100)/len(test_pos))
		print("Our accuracy :", obs.accuracy)

		for cnt, k in enumerate(tags2):
			per_tag_pred = np.array([1 if x == k else 0 for x in out_tags1])
			per_tag_crkt = np.array([1 if x == k else 0 for x in test_pos])
			obs.per_pos_acc[k] = (sum(per_tag_pred == per_tag_crkt) * 100) / len(test_pos)
			conf = np.zeros((2, 2))
			conf[0][0] = sum(per_tag_crkt * per_tag_pred)
			conf[0][1] = sum((1 - per_tag_crkt) * per_tag_pred)
			conf[1][0] = sum(per_tag_crkt * (1 - per_tag_pred))
			conf[1][1] = sum((1 - per_tag_crkt) * (1 - per_tag_pred))
			obs.conf_matx[k] = conf
			obs.conf = confusion_matrix(test_pos, out_tags1, labels=tags2)
			obs.labels = tags2

		observations.append(obs)

	print("Finding accumulated results...")

	train_avg_acc = 0
	for obsv in observations:
		train_avg_acc += obsv.train_accuracy

	print("Our average training accuracy: ", train_avg_acc/num_cross_valid)
	
	avg_acc = 0
	for obsv in observations:
		avg_acc += obsv.accuracy

	print("Our average test accuracy: ", avg_acc/num_cross_valid)

	per_pos = dict()
	count = dict()
	for kk in tags:
		per_pos[kk]=0
		count[kk]=0
		for k in observations:
			if kk in k.per_pos_acc:
				per_pos[kk]+=k.per_pos_acc[kk]
				count[kk]+=1

	for tg, vl in per_pos.items():
		if vl!=0:
			per_pos[tg] = per_pos[tg]/count[tg]
	print("per-POS accuracy :")
	print(per_pos)

	cnt=0
	conf_mt = dict()
	for i in observations:
		for tg, vl in i.conf_matx.items():
			if tg not in conf_mt:
				conf_mt[tg]=vl
			else:
				conf_mt[tg]+=vl
		cnt+=1

	for tg, _ in conf_mt.items():
		conf_mt[tg] /= cnt

	print("Confusion Matrix (o vs r) :")
	print(conf_mt)

	fig = plt.figure(figsize = (12, 12))
	sn.heatmap(observations[0].conf, annot=False, cmap="YlGnBu")
	fig.suptitle('Confusion Matrix')
	plt.xlabel('Tags')
	plt.ylabel('Tags')
	ind = [j for j in range(len(observations[0].labels))]
	tgs = observations[0].labels
	plt.xticks(ind, tgs)
	plt.yticks(ind, tgs)
	plt.show()


def cvxopt_train(X, y):
	C = 2
	m,n = X.shape
	y = y.reshape(-1,1) * 1.
	X_dash = y * X
	H = np.dot(X_dash , X_dash.T) * 1.

	#Converting into cvxopt format 
	P = cvxopt_matrix(H)
	q = cvxopt_matrix(-np.ones((m, 1)))
	G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
	h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
	A = cvxopt_matrix(y.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))

	#Run solver
	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])

	w = ((y * alphas).T @ X).reshape(-1,1)
	S = (alphas > 1e-4).flatten()
	b = y[S] - np.dot(X[S], w)
	return w, b

def cvxopt_predict(w, b, x_test):
	y_pred = []
	for i in x_test:
		pred = np.dot(w.T,i) + b
		if(pred[0][0] > 0):
			y_pred.append(np.array([0, 1]))
		elif(pred[0][0] < 0):
			y_pred.append(np.array([1, 0]))
	return y_pred

if __name__ == '__main__':
	importdata()

	# Run only once
	print("Finding word embedding model...")
	model = get_word_embedding_model()
	word_vect = model.wv
	# print(word_vect['the'])
	
	# print("Loading word embedding model...")
	# fname = get_tmpfile("word_vector.kv")
	# word_vect.save(fname)
	# word_vect = KeyedVectors.load(fname, mmap='r')
	# print(word_vect['the'])
	print("Begin...")
	svm(5, word_vect)
	print("End.")
