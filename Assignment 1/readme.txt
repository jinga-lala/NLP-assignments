#SVM:
Requirements -
Python3
   + 
pip install -U scikit-learn
pip install gensim
pip install nltk
pip install cvxopt

(gensim for word2vec, cvxopt for quadratic optimization solver reqiured for svm).

Execution instructions:
python3 svm.py

(Detailed output would show the steps passed while running)

#Bi-LSTM:
Requirements
pip3 install nltk
pip3 install numpy
pip3 install seaborn
pip3 install matplotlib
pip3 install sklearn
pip3 install tqdm
pip3 install torch

Execution instruction
python3 bilstm.py --cross_validation 0 --load_model 0 #trains for one split
python3 bilstm.py --cross_validation 1 --load_model 0 #trains for 5-fold cross validation
python3 bilstm.py --cross_validation 0 --load_model 1 #loads a pretrained model and test its performance