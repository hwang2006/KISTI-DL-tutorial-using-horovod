import numpy as np
import random
import time
import datetime
import pandas as pd
import os

import urllib.request
#from Korpora import Korpora


from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import argparse
from argparse import ArgumentParser
from tqdm import tqdm

#MAX_LEN = 128
#BATCH_SIZE = 32
verbose = 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NSMC Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/scratch/qualis/ptl/ratings_train.txt'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/scratch/qualis/ptl/ratings_test.txt'),
                    help='path to validation data')
parser.add_argument('--pretrained-model', default='beomi/kcbert-large',
                    help='Transformers PLM name')
parser.add_argument('--max-length', type=int, default=128,
                    help='the maximum length of a tokenized sentence')
parser.add_argument('--log-dir', default='./pt_logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

#Korpora.fetch(
#    corpus_name="nsmc",
#    root_dir=".",
#    force_download=True,
#)

def create_dataloader(df, tokenizer):
    #df['document'].nunique() 
    #df['label'].nunique()
    df.drop_duplicates(subset=['document'], inplace=True)
    df = df.dropna(how = 'any')   	

    # Create Train data loader
    sentences = df['document']
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = df['label'].values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=args.max_length, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    df_inputs = torch.tensor(input_ids)
    df_labels = torch.tensor(labels)
    df_masks = torch.tensor(attention_masks)

    df_data = TensorDataset(df_inputs, df_masks, df_labels)
    df_sampler = RandomSampler(df_data)
    df_dataloader = DataLoader(df_data, sampler=df_sampler, batch_size=args.batch_size)
    return df_dataloader
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    #convert to hh:mm:ss 
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(epoch):
  #print("")
  #print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
  #print('Training...')

  t0 = time.time()
  total_loss = 0

  model.train()
  with tqdm(total=len(train_dataloader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
    for step, batch in tqdm(enumerate(train_dataloader)):
        #if step % 500 == 0 and not step == 0:
        #    elapsed = format_time(time.time() - t0)
        #    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch
        outputs = model(input_ids,
                        token_type_ids=None,
                        attention_mask=input_mask,
                        labels=labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        # gradiant clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        #model.zero_grad()
        t.set_postfix({'loss': total_loss/(step+1)})
        t.update(1)

  avg_train_loss = total_loss / len(train_dataloader)

  print("Average training loss: {0:.2f}".format(avg_train_loss))
  print("Training epcoh took: {:}\n".format(format_time(time.time() - t0)))

def test(epoch):
# Start testing 
  t0 = time.time()
  total_loss = 0

  correct = 0
  total_correct = 0

  #eval_loss, eval_accuracy = 0, 0
  acc_accuracy = 0
  nb_eval_steps, nb_eval_examples = 0, 0

  model.eval()

  with tqdm(total=len(test_dataloader),
			  desc='Test Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
    for step, batch in enumerate(test_dataloader):
        #if step % 500 == 0 and not step == 0:
        #  elapsed = format_time(time.time() - t0)
        #  print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch
        n_ids = len(input_ids)
    
        with torch.no_grad():     
          outputs = model(input_ids, 
                        token_type_ids=None, 
                        attention_mask=input_mask,
					    labels=labels)
    
        loss = outputs[0]
        total_loss += loss.item()
        logits = outputs[1]

        #logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().item()
        total_correct += correct

        current_batch_accuracy = correct/n_ids
        acc_accuracy += current_batch_accuracy
    
        #tmp_eval_accuracy = flat_accuracy(logits, labels)
        #eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

        t.set_postfix({'loss': total_loss/ nb_eval_steps, 
					   #'Accuracy': step_accuracy})
					   'Accuracy': acc_accuracy/nb_eval_steps})
        t.update(1)

  print("Accuracy: {0:.2f}".format(total_correct/len(test_dataloader.dataset)))
  print("Test took: {:}\n".format(format_time(time.time() - t0)))

if __name__ == '__main__':

  seed_val = 42
  random.seed(seed_val)
  #np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  allreduce_batch_size = args.batch_size * args.batches_per_allreduce

  args.cuda = not args.no_cuda and torch.cuda.is_available()
  if args.cuda:
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
  else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

  train_data = pd.read_table(args.train_dir)
  test_data = pd.read_table(args.val_dir)

  tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model,
    do_lower_case=False,
  )

  train_dataloader = create_dataloader(train_data, tokenizer)
  test_dataloader = create_dataloader(test_data, tokenizer) 

  #model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2)
  model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
  model = model.to(device)

  optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

  total_steps = len(train_dataloader) * args.epochs

  scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

  #optimizer = optim.Adadelta(model.parameters(), lr=2e-5)
  #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  #scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    #save_chechpoint(epoch)
     
print("")
print("Training and Testing complete!")
