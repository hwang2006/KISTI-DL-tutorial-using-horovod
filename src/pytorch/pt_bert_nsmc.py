#import numpy as np
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
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import argparse
from argparse import ArgumentParser
from tqdm import tqdm

import socket
import horovod.torch as hvd

#MAX_LEN = 128
#BATCH_SIZE = 32
#verbose = 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NSMC Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--train-dir', default=os.path.expanduser('/scratch/qualis/ptl/ratings_train.txt'),
#                    help='path to training data')
parser.add_argument('--train-dir', default=os.path.expanduser('./ratings_train.txt'),
                    help='path to training data')
#parser.add_argument('--test-dir', default=os.path.expanduser('/scratch/qualis/ptl/ratings_test.txt'),
#                    help='path to validation data')
parser.add_argument('--test-dir', default=os.path.expanduser('./ratings_test.txt'),
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
parser.add_argument('--test-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=2e-5,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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

def create_dataloader(df, kwargs, tokenizer):
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
    #df_sampler = RandomSampler(df_data)
    df_sampler = torch.utils.data.distributed.DistributedSampler(
        df_data, num_replicas=hvd.size(), rank=hvd.rank())
    #df_dataloader = DataLoader(df_data, sampler=df_sampler, batch_size=args.batch_size)
    df_dataloader = DataLoader(df_data, sampler=df_sampler, **kwargs)
    return df_dataloader, df_sampler
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    #convert to hh:mm:ss 
    return str(datetime.timedelta(seconds=elapsed_rounded))

'''
def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n
'''
def save_checkpoint(epoch):
    if hvd.rank() == 0:
       filepath = args.checkpoint_format.format(epoch=epoch + 1)
       state = {
           'model': model.state_dict(),
           'optimizer': optimizer.state_dict(),
       }
       torch.save(state, filepath)


def train(epoch):
  #print("")
  #print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
  #print('Training...')

  t0 = time.time()
  total_loss = 0

  model.train()
  train_sampler.set_epoch(epoch)
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

  #avg_train_loss = total_loss / len(train_dataloader)
  avg_train_loss = total_loss / len(train_dataloader)

  #if log_writer:
        #log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        #log_writer.add_scalar('train/loss', avg_train_loss, epoch)
        #log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        #log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

  if (verbose):
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
  nb_eval_steps, nb_test_data = 0, 0

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
        nb_test_data +=n_ids

        t.set_postfix({'loss': total_loss/ nb_eval_steps, 
					   #'Accuracy': step_accuracy})
					   'Accuracy': acc_accuracy/nb_eval_steps})
        t.update(1)
  
  #if log_writer:
        #log_writer.add_scalar('test/loss', val_loss.avg, epoch)
        #log_writer.add_scalar('test/loss', total_loss/ nb_eval_steps, epoch)
        #log_writer.add_scalar('test/accuracy', val_accuracy.avg, epoch)
        #log_writer.add_scalar('test/accuracy', acc_accuracy/nb_eval_steps, epoch)

  if(verbose):
    #print("Accuracy: {0:.3f}".format(total_correct/len(test_dataloader.dataset)))
    print("Accuracy: {0:.2f}".format(total_correct/nb_test_data))
    print("Test took: {:}\n".format(format_time(time.time() - t0)))

if __name__ == '__main__':

  args = parser.parse_args()
  #seed_val = 42
  random.seed(args.seed)
  #np.random.seed(seed_val)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  allreduce_batch_size = args.batch_size * args.batches_per_allreduce

  hvd.init()
  print('************* hvd.size:', hvd.size(),'hvd.rank:', hvd.rank(),\
      'hvd.local_rank:', hvd.local_rank(), 'hostname:', socket.gethostname())


  if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

  if args.cuda:
        device = torch.device("cuda")
  else:
        device = torch.device("cpu") 


  cudnn.benchmark = True

  # If set > 0, will resume training from a given checkpoint.
  resume_from_epoch = 0
  for try_epoch in range(args.epochs, 0, -1):
      if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
          resume_from_epoch = try_epoch
          break

  # Horovod: broadcast resume_from_epoch from rank 0 (which will have
  # checkpoints) to other ranks.
  resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                    name='resume_from_epoch').item()

  # Horovod: print logs on the first worker.
  verbose = 1 if hvd.rank() == 0 else 0

  # Horovod: write TensorBoard logs on first worker.
  log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None


  train_data = pd.read_table(args.train_dir)
  test_data = pd.read_table(args.test_dir)

  #train_kwargs = {'batch_size': args.batch_size, 'shuffle' : True}
  train_kwargs = {'batch_size': args.batch_size}
  test_kwargs = {'batch_size': args.test_batch_size}

  #if args.cuda:
       #cuda_kwargs = {'num_workers': 1,
       #               'pin_memory': True,
       #               'shuffle': True}
  #     cuda_kwargs = {'num_workers': 4,
  #                    'pin_memory': True}
  #     train_kwargs.update(cuda_kwargs)
  #     test_kwargs.update(cuda_kwargs)
 

  # Horovod: limit # of CPU threads to be used per worker.
  torch.set_num_threads(4)

  cuda_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)  

  # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
  # issues with Infiniband implementations that are not fork-safe
  if (cuda_kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
          mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
      cuda_kwargs['multiprocessing_context'] = 'forkserver'

  tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model,
    do_lower_case=False,
  )

  train_dataloader, train_sampler = create_dataloader(train_data, train_kwargs, tokenizer)
  test_dataloader, test_sampler = create_dataloader(test_data, test_kwargs, tokenizer) 
  #print(len(test_sampler.dataset)) 
  #49157
  #print(len(test_dataloader.dataset)) 
  #49157

  #model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2)
  model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
  model = model.to(device)

  # By default, Adasum doesn't need scaling up learning rate.
  # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
  lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

  # If using GPU Adasum allreduce, scale learning rate by local_size.
  if args.use_adasum and hvd.nccl_built():
       lr_scaler = args.batches_per_allreduce * hvd.local_size()


  optimizer = AdamW(model.parameters(),
                  #lr = 2e-5,
                  lr = (args.base_lr * lr_scaler),
                  eps = 1e-8
                )

  # Horovod: (optional) compression algorithm.
  compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

  # Horovod: wrap optimizer with DistributedOptimizer.
  optimizer = hvd.DistributedOptimizer(
      optimizer, named_parameters=model.named_parameters(),
      compression=compression,
      backward_passes_per_step=args.batches_per_allreduce,
      op=hvd.Adasum if args.use_adasum else hvd.Average,
      gradient_predivide_factor=args.gradient_predivide_factor)

  # Horovod: wrap optimizer with DistributedOptimizer.
  #optimizer = hvd.DistributedOptimizer(optimizer,
  #                                       named_parameters=model.named_parameters(),
  #										 op=hvd.Average)

  total_steps = len(train_dataloader) * args.epochs
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
  
  #optimizer = optim.Adadelta(model.parameters(), lr=2e-5)
  #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  #scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

  # Restore from a previous checkpoint, if initial_epoch is specified.
  # Horovod: restore on the first worker which will broadcast weights to other workers.
  if resume_from_epoch > 0 and hvd.rank() == 0:
      filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
      checkpoint = torch.load(filepath)
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])

  # Horovod: broadcast parameters & optimizer state. 
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)

for epoch in range(resume_from_epoch, args.epochs):
    #train(epoch, train_sampler)
    train(epoch)
    test(epoch)
    save_checkpoint(epoch)
     
print("")
print("Training and Testing complete!")
