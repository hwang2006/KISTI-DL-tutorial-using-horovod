'''
1) Single node, multi-GPUs
$ python pt_mnist_lightning.py

2) 2 nodes, 2 GPUs
$ srun -N 2 --ntasks-per-node=2 python pt_mnist_lightning.py --num_nodes 2
'''

import os
#import pandas as pd
#import seaborn as sn
#from IPython.display import display

# Pytorch modules
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split


# Pytorch-Lightning
from lightning import LightningDataModule, LightningModule, Trainer
#from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.loggers import CSVLogger
import lightning as L 


# Dataset
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import transforms

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class LitMNIST(LightningModule):
    def __init__(self):
        super(LitMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        #self.learning_rate = 1.0
        self.learning_rate = 1e-3
        self.data_dir = PATH_DATASETS
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


from argparse import ArgumentParser

def main(hparams):
    model = LitMNIST()
    trainer = L.Trainer(
        #accelerator="cpu",
        #devices=1,
        #strategy="ddp_spawn",
        #accelerator="gpu",
        #accelerator="auto",
        accelerator=hparams.accelerator,
        strategy=hparams.strategy,
        devices=hparams.devices,
        max_epochs=10,
        num_nodes=hparams.num_nodes,
        logger=CSVLogger(save_dir="logs/"),
    )
    trainer.fit(model)

    # Lightning will automatically test using the best saved checkpoint (conditioned on val_loss)
    trainer.test()
    
    #metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    #del metrics["step"]
    #metrics.set_index("epoch", inplace=True)
    #display(metrics.dropna(axis=1, how="all").head())
    #sn.relplot(data=metrics, kind="line")

if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "auto")
    parser.add_argument("--devices", default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
    parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else "auto")
    parser.add_argument("--num_nodes", default=1)
    args = parser.parse_args()

    main(args)

