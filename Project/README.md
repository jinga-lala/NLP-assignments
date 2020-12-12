# README
# Toxic Comment Classification
# CS626 NLP course project

Code implemented by the following team members:
- Yash Jain, 170050055
- Debarbata Mandal, 170050073
- Aditya Sharma, 170050043


The submission directory contains the following files:
```
submission
│  README.md // current file
│  requirements.txt  // python packages required to run the code
│  main.py // main file where the code resides
│  models // trained bert model
│  Transformer.ipynb // sample notebook used for experiments with comments for easy understanding
│  BASELINE_LSTM.ipynb // notebook running the baseline LSTM model
│  report.pdf // report
|  slides.pdf // slides
|  submission.csv // our best submission in this competition using BERT
```

#### Instructions to run the code
Please use Python3.7+ to run the code.
The arguments to `main.py` are:
```
  --model MODEL                         Pre-trained transformer model that needs to be used
  --epochs EPOCHS                       Number of training epochs
  --train_data TRAIN_DATA               Path to train data
  --test_data TEST_DATA                 Path to test data
  --test_label_data TEST_LABEL_DATA     Path to test label data
  --batch_size BATCH_SIZE               Batch Size
  --max_seq_len MAX_SEQ_LEN             Maximum length of an input sentence
  --epochs EPOCHS                       Number of training epochs
  --lr LR                               Learning rate for student training
  --cuda                                Use gpu
```
cd into the submission directory and run the following commands: 
```sh
$ pip install -r requirements.txt
$ python3 main.py --model "bert"  --epochs 1 --train_data <train_data path> --test_data <test_data path> --test_label_data <test_label_data path>
```
