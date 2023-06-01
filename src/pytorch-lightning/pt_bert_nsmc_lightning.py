import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
#from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

class Arg:
    random_seed: int = 42  # Random Seed
    pretrained_model: str = 'beomi/kcbert-large'  # Transformers PLM name
    pretrained_tokenizer: str = ''  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    auto_batch_size: str = 'power'  # Let PyTorch Lightening find the best batch size 
    batch_size: int = 0  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size` 
    lr: float = 5e-6  # Starting Learning Rate
    epochs: int = 3  # Max Epochs
    max_length: int = 150  # Max Length input size
    report_cycle: int = 100  # Report (Train Metrics) Cycle
    train_data_path: str = "nsmc/ratings_train.txt"  # Train Dataset file 
    val_data_path: str = "nsmc/ratings_test.txt"  # Validation Dataset file 
    #cpu_workers: int = os.cpu_count()  # Multi cpu workers
    cpu_workers: int = 8  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores

args = Arg()

# args.tpu_cores = 8  # Enables TPU
args.fp16 = True  # Enables GPU FP16
args.batch_size = 16  # Force setup batch_size

class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertForSequenceClassification.from_pretrained(self.args.pretrained_model)
        self.validation_step_outputs = []
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits
        
        preds = logits.argmax(dim=-1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'train_loss': loss.cpu().detach().numpy().tolist(),
            'train_acc': metrics[0],
            'train_precision': metrics[1],
            'train_recall': metrics[2],
            'train_f1': metrics[3],
        }
        if (batch_idx % self.args.report_cycle) == 0:
            print()
            pprint(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
   
        self.validation_step_outputs.append({
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        })

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def on_validation_epoch_end(self):
        loss = torch.tensor(0, dtype=torch.float)
        for i in self.validation_step_outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(self.validation_step_outputs)

        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in self.validation_step_outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'val_loss': loss,
            'val_acc': metrics[0],
            'val_precision': metrics[1],
            'val_recall': metrics[2],
            'val_f1': metrics[3],
        }

        print()
        pprint(tensorboard_logs)
        self.validation_step_outputs.clear()
  
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df

    def train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def val_dataloader(self):
        df = self.read_data(self.args.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )


def main(cliargs):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)
    model = Model(args)

    print(":: Start Training ::")
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        #auto_scale_batch_size=args.auto_batch_size if args.auto_batch_size and not args.batch_size else False,
        # For GPU Setup
        #deterministic=torch.cuda.is_available(),
        accelerator=cliargs.accelerator,
        strategy=cliargs.strategy,
        devices=cliargs.devices,
        num_nodes = cliargs.num_nodes,
        #gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
        logger=CSVLogger(save_dir="logs/")
    )
    trainer.fit(model)


from argparse import ArgumentParser

if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "auto")
    #parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
    #parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else "auto")
    parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else None)
    parser.add_argument("--num_nodes", default=1)
    cliargs = parser.parse_args()

    main(cliargs)

