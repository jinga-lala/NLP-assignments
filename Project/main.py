import numpy as np 
import pandas as pd 
import os

import pandas as pd, numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from cleantext import clean
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import accuracy_score
import argparse


parser = argparse.ArgumentParser(description="Argument Parsers for CS626 Course Project")
parser.add_argument("--model", action="store", dest="model", default="bert", type=str, required=False, help="Pre-trained transformer model that needs to be used")
parser.add_argument("--train_data", action="store", type=str, default=None, help="Path to train data")
parser.add_argument("--test_data", action="store", type=str, default=None, help="Path to test data")
parser.add_argument("--test_label_data", action="store", type=str, default=None, help="Path to test label data")
parser.add_argument("--epochs", action="store", dest="epochs", type=int, default=2, help="Number of  training epochs")
parser.add_argument("--batch_size", action="store", dest="batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--max_seq_len", action="store", dest="max_seq_len", type=int, default=128, help="Maximum length of an input sentence")
parser.add_argument("--lr", action="store", dest="lr", type=float, default=5e-3, help="Learning rate for training")
parser.add_argument("--cuda", action="store_true", dest="cuda", help="Use cuda")

args = parser.parse_args()
# **Load the training dataset in a pandas dataframe**
if args.train_data is None:
    train_path = '../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
else:
    train_path = args.train_data
train = pd.read_csv(train_path)

# **Define the output classes**

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[classes].values


# **Perform dataset cleaning such as removing stop words, emojis, punctuations etc. using clean-text library**


def clean_string(s): return clean(s, no_line_breaks=True, no_urls=True, no_punct=True)

# **Custom Dataset loader**

class MyData(torch.utils.data.Dataset):
    def __init__(self, data, label_cols):
        self.data = data
        self.label_cols = label_cols

    def __getitem__(self, item):
        comment = clean_string(self.data.comment_text[item])
        toxic = self.data.toxic[item]
        severe_toxic = self.data.severe_toxic[item]
        obscene = self.data.obscene[item]
        threat = self.data.threat[item]
        insult = self.data.insult[item]
        identity_hate = self.data.identity_hate[item]
        return comment, torch.Tensor([toxic, severe_toxic, obscene, threat, insult, identity_hate])
    
    def __len__(self):
        return len(self.data)


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'comment_text'
label_cols.append(COMMENT)

# **Set device for PyTorch Training**
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
print(device)

# **Model parameters and hyper-parameters**
MAX_SEQ_LEN = args.max_seq_len
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
if args.model=='bert':
    model_name = 'bert-base-uncased'
elif args.model=='electra':
    model_name = 'google/electra-small-discriminator'
elif args.model=='gpt':
    model_name = 'gpt2'#'bert-base-uncased'
lstm_units = 50

# **Define the tokenizer.**
# This is dependent on the transformer model passed as input in model_name. Also specify the max sentence length after tokenizing. We are using BERT Word-piece tokenizer for LSTM baseline to avoid any performance disadvantage due to the tokenizer.

import transformers
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MAX_SEQ_LEN, do_lower_case=True, add_special_tokens=True,
                                                max_length=MAX_SEQ_LEN, pad_to_max_length=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# **Load the training data into torch Dataset and perform train-validation set split.**

train_set = MyData(train, label_cols)
train_set, val_set = torch.utils.data.random_split(train_set, [int(0.9*len(train_set)), len(train_set)-int(0.9*len(train_set))] )
train_set = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_set = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# **Define custom nn model**

class AutoNet(nn.Module):
  def __init__(self, seqLength=MAX_SEQ_LEN, numClasses=6, model_name=model_name, lstm_units=50):
    super(AutoNet, self).__init__()
    if model_name=='google/electra-small-discriminator':
        self.lstm = nn.LSTM(256, lstm_units, dropout=0.1,
                        num_layers=1, bidirectional=True, batch_first=True, bias=False)
    else:
        self.lstm = nn.LSTM(768, lstm_units, dropout=0.1,
                        num_layers=1, bidirectional=True, batch_first=True)
    self.model = nn.Embedding(len(tokenizer), 768)
    self.l1 = nn.Linear(lstm_units * 2, 50)
    self.d1 = nn.Dropout(0.2)
    self.l2 = nn.Linear(50, numClasses)
    self.sigmoid = nn.Sigmoid()

  def forward(self,input_ids, attention_mask=None):
    x = self.model(input_ids)
    x, (hidden, cell) = self.lstm(x)
    x,_ = torch.max(x, dim=1)
    x = self.l1(x)
    x = self.d1(x)
    x = self.l2(x)
    x = self.sigmoid(x)
    return x


# **Initialize the model, optimizer and the loss function used.**

model = AutoNet()
model = model.to(device)
model.train()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_criteria = nn.BCELoss()
loss_criteria = loss_criteria.to(device)
from torchsummary import summary
print(summary(model))


# **Training the Model**

model.train()
from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
  count = 0
  total_loss =0 
  model.train()
  for i,data in enumerate(train_set):
    optimizer.zero_grad()
    enc = tokenizer.batch_encode_plus(list(data[0]), pad_to_max_length=True, max_length=MAX_SEQ_LEN, 
                                return_tensors='pt', add_special_tokens=True, return_attention_mask=True,
                                return_token_type_ids=False, )
    input_ids = enc['input_ids'].to(device)
    labels = torch.tensor(data[1]).to(device)
    out = model(input_ids=input_ids) 
    loss = loss_criteria(out, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.detach().data
    if (i+1)%256==0:
        print(f"Epoch: {epoch}, batch: {i+1}, loss: {total_loss/(BATCH_SIZE)}")
        total_loss = 0

  model.eval()
  all_pred = []
  all_gold = []
  with torch.no_grad():
      for i,data in enumerate(val_set):
        enc = tokenizer.batch_encode_plus(list(data[0]), pad_to_max_length=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        labels = torch.tensor(data[1]).to(device)
        out = model(input_ids=input_ids)
        all_pred.extend(1*(out>0.98).clone().detach().cpu().numpy())
        all_gold.extend((labels.type(torch.LongTensor).detach().cpu().numpy()))

  count=0
  for i in range(len(all_gold)):
    if (all_gold[i]==all_pred[i]).all():
      count+=1
  print("Validation accuracy:", count/len(all_gold))


# **Code cell for demo and some error analysis**

sample_text = "Please get out of here"
clean_txt = clean_string(sample_text)
out1 = tokenizer([clean_txt], pad_to_max_length=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
input_ids = out1['input_ids'].to(device)
attention_mask = out1['attention_mask'].to(device)
model.eval()
with torch.no_grad():
    preds = model(input_ids=input_ids, attention_mask=attention_mask)
print(preds)


# **Load Test data and analyse**

if args.test_data is None:
    test = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'
else:
    test = args.test_data
test = pd.read_csv(test)
if args.test_label_data is None:
    test_labels_path = '../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'
else:
    test_labels_path = args.test_label_data
    
test_labels = pd.read_csv(test_labels_path)
test_labels = test_labels.replace(to_replace=-1,value=0)
# test_labels.sample(20)
test_set = test.merge(test_labels, left_index=True, right_index=True)
test_set = test_set[["id_x", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
test_set = test_set.reset_index(drop=True)
test_set = test_set.rename(columns={"id_x": "id"})

# **Perform test data cleaning and load into Dataloader**

test_set['comment_text'] = test_set['comment_text'].apply(clean_string)
ids = test_set['id']
test_set = MyData(test_set, classes)
test_set = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# **Model evaluation**

model.eval()
all_pred = []
all_gold = []
with torch.no_grad():
    for i,data in enumerate(test_set):
        enc = tokenizer.batch_encode_plus(list(data[0]), pad_to_max_length=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        labels = torch.tensor(data[1]).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        all_pred.extend(out.clone().detach().cpu().numpy())
        all_gold.extend((labels.type(torch.LongTensor).detach().cpu().numpy()))

print("Test accuracy:", accuracy_score(all_gold, (np.array(all_pred) > 0.98)))


# **Create CSV file for Kaggle Submission**

target_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
ids = pd.Series(ids)
y_preds = pd.DataFrame(all_pred, columns=target_columns)
final_submission = pd.concat([ids, y_preds], axis=1)
final_submission.head()
final_submission.to_csv('submission.csv', index=False)
