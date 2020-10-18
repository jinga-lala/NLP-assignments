import itertools
from itertools import chain
import matplotlib.pyplot as plt
import nltk
import sklearn
import seaborn as sn
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix

# Feature generation function
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'b': 1.0,
        'word': word.lower(),
        'suff1': word[-3:],
        'suff2': word[-2:],
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'pos': postag,        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            'prv_word': word1.lower(),
            'prv_word_is_title': word1.istitle(),
            'prv_word_is_upper': word1.isupper(),
            'prv_pos': postag1,
        })
    else:
        features['is_beginning'] = True
        
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.update({
            'p_p_word': word1.lower(),
            'p_p_word_is_title': word1.istitle(),
            'p_p_word_is_upper': word1.isupper(),
            'p_p_pos': postag1,
        })
    else:
        features['is_first_wrd'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label.split('-')[0] for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

#load dataset
trainf = open("./assignment2dataset/train.txt")
testf = open("./assignment2dataset/test.txt")

train_sents = []
test_sents = []

sent = []
for line in trainf.readlines():
    if len(line.split()) == 0:
            train_sents.append(sent)
            sent=[]
    elif line.split()[-1] != 'O':
        sent.append(line.split())

sent = []
for line in testf.readlines():
    if len(line.split()) == 0:
            test_sents.append(sent)
            sent=[]
    elif line.split()[-1] != 'O':
        sent.append(line.split())
        
# train_sents[0]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
# labels


# In[67]:


y_pred = crf.predict(X_test)
overall_f1 = metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)
overall_prec = metrics.flat_precision_score(y_test, y_pred, 
                      average='weighted', labels=labels)
overall_recall = metrics.flat_recall_score(y_test, y_pred, 
                      average='weighted', labels=labels)

print("Overall F1:", overall_f1) 
print("Overall Precision:", overall_prec)
print("Overall Recall", overall_recall)

# Inspect per-class results in more detail:

print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
))

y_test_merged = list(itertools.chain(*y_test))
y_pred_merged = list(itertools.chain(*y_pred))
# print(type(y_test), y_test[0:20], y_pred[0:20])
conf = confusion_matrix(y_test_merged, y_pred_merged, labels=labels)

# ax= plt.subplot()
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(['B', 'I'])
# ax.yaxis.set_ticklabels(['B', 'I'])
# sns.heatmap(conf, annot=True, ax = ax)
# input("Press any key to close")

fig = plt.figure(figsize = (4, 4))
sn.heatmap(conf, annot=False)
fig.suptitle('Confusion Matrix')
plt.xlabel('Tags')
plt.ylabel('Tags')
ind = [j for j in range(2)]
tgs = labels
plt.xticks(ind, tgs)
plt.yticks(ind, tgs)
plt.show()
