import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import socket
from tqdm import tqdm
import horovod.torch as hvd

# Horovod: initialize library.
hvd.init()
torch.manual_seed(42)

print('************* hvd.size:', hvd.size(),'hvd.rank:', hvd.rank(),\
        'hvd.local_rank:', hvd.local_rank(), 'hostname:', socket.gethostname())

# Horovod: pin GPU to local rank.
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(42)

#batch_size= 1
batch_size = 32
learning_rate = 0.0002
#learning_rate = 0.002
num_epoch = 100 

# 라벨(혹은 클래스) 별로 폴더가 저장되어 있는 루트 디렉토리를 지정합니다.
#img_dir = "./images"
train_dir = "/scratch/qualis/dataset/training_set/training_set"
valid_dir = "/scratch/qualis/dataset/test_set/test_set"

# 해당 루트 디렉토리를 ImageFolder 함수에 전달합니다.
# 이때 이미지들에 대한 변형도 같이 전달해줍니다.
train_data = dset.ImageFolder(train_dir, transforms.Compose([
                                      transforms.Resize(256),                   
                                      transforms.RandomResizedCrop(224),       
                                      transforms.RandomHorizontalFlip(),    
                                      transforms.ToTensor(),  
            ]))

valid_data = dset.ImageFolder(valid_dir, transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
            ]))


#train_loader = data.DataLoader(train_data, batch_size=batch_size,
#                            shuffle=True, num_workers=2)

#valid_loader = data.DataLoader(valid_data, batch_size=batch_size,
#                            shuffle=True, num_workers=2)


train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data,
    num_replicas=hvd.size(),
    rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    num_workers=4,
    sampler=train_sampler)


valid_sampler = torch.utils.data.distributed.DistributedSampler(
    valid_data,
    num_replicas=hvd.size(),
    rank=hvd.rank())
valid_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=32,
    num_workers=4,
    sampler=valid_sampler)



def conv_block_1(in_dim,out_dim,act_fn,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=stride),
        act_fn,
    )
    return model


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model


class BottleNeck(nn.Module): 
    def __init__(self,in_dim,mid_dim,out_dim,act_fn,down=False):
        super(BottleNeck,self).__init__()
        self.down=down
        
        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
              conv_block_1(in_dim,mid_dim,act_fn,2), # strid = 2 ==> 이미지 사이지 반으로 줄임 
              conv_block_3(mid_dim,mid_dim,act_fn),
              conv_block_1(mid_dim,out_dim,act_fn),
            )
            self.downsample = nn.Conv2d(in_dim,out_dim,1,2)
            
        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim,mid_dim,act_fn),
                conv_block_3(mid_dim,mid_dim,act_fn),
                conv_block_1(mid_dim,out_dim,act_fn),
            )
            
        # 더하기를 위해 차원을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim,out_dim,kernel_size=1)
                  
    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out


class ResNet(nn.Module): #resnet.png 참조

    def __init__(self, base_dim, num_classes=2):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim,base_dim,base_dim*4,self.act_fn),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.act_fn),
            #BottleNeck(base_dim*4,base_dim,base_dim*4,self.act_fn,down=True),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.act_fn),
        )   
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4,base_dim*2,base_dim*8,self.act_fn, down=True),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.act_fn),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.act_fn),
            #BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.act_fn,down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8,base_dim*4,base_dim*16,self.act_fn, down=True),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.act_fn),            
            #BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            #BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            #BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.act_fn,down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16,base_dim*8,base_dim*32,self.act_fn, down=True),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.act_fn),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.act_fn),
        )
        self.avgpool = nn.AvgPool2d(7,1) 
        self.fc_layer = nn.Linear(base_dim*32,num_classes)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        #out = out.view(batch_size,-1)
        out = out.view(out.size(0),-1)
        out = self.fc_layer(out)
        
        return out

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def test():
   model.eval()
   val_loss = 0.
   val_accuracy = 0.
   for image,label in valid_loader:
      #x = image.to(device)
      x = image.cuda()
      #y_= label.to(device)
      y_= label.cuda()

      #output = model.forward(x)
      output = model(x)
      #print(y_, output)
      #_,output_index = torch.max(output,1)

      #total += label.size(0)
      #correct += (output_index == y_).sum().float()

      val_loss += F.nll_loss(output, y_, size_average=False).item()
      #val_loss += loss_func(output, y_)
      #print(val_loss)
      #val_loss += F.nll_loss(output, y_).item()
      pred = output.data.max(1, keepdim=True)[1]
      #val_accuracy += pred.eq(y_.data.view_as(pred)).cpu().float().sum()
      val_accuracy += pred.eq(y_.data.view_as(pred)).float().sum()

   # Horovod: use test_sampler to determine the number of examples in
   # this worker's partition.
   val_loss /= len(valid_sampler)
   val_accuracy /= len(valid_sampler)
   # Horovod: average metric values across workers.
   val_loss = metric_average(val_loss, 'avg_loss')
   val_accuracy = metric_average(val_accuracy, 'avg_accuracy')

   # Horovod: print output only on first rank.
   if hvd.rank() == 0:
       print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
          val_loss, 100. * val_accuracy))


#model = ResNet(base_dim=64).to(device)
model = ResNet(base_dim=64).cuda()

loss_func = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=learning_rate*hvd.size())
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=model.named_parameters())

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

#for i in model.children():
#    print(i)

for i in range(num_epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(num_epoch)
    for j,[image,label] in enumerate(train_loader) :
        #x = image.to(device)
        x = image.cuda()
        #y_= label.to(device)
        y_= label.cuda()
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    #if i % 5 ==0:
        #print(loss)
        if j % 20 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i+1, j * len(x), len(train_sampler),
                100. * j / len(train_loader), loss.item() ))
    test()

correct = 0
total = 0

with torch.no_grad():
  for image,label in valid_loader:
      #x = image.to(device)
      x = image.cuda()
      #y_= label.to(device)
      y_= label.cuda()

      output = model.forward(x)
      #print(y_, output)
      _,output_index = torch.max(output,1)

      total += label.size(0)
      correct += (output_index == y_).sum().float()

  print("Accuracy of Test Data: {}".format(100*correct/total))
  # Horovod: average metric values across workers.
  #correct = metric_average(correct, 'avg_correct')
  #print("Average Accuracy of Test Data: {}".format(100*correct/total))

'''
val_loss = 0.
val_accuracy = 0.

with torch.no_grad():
  for image,label in tqdm(valid_loader):
      #x = image.to(device)
      x = image.cuda()
      #y_= label.to(device)
      y_= label.cuda()

      #output = model.forward(x)
      output = model(x)
      #print(y_, output)
      #_,output_index = torch.max(output,1)

      #total += label.size(0)
      #correct += (output_index == y_).sum().float()

      val_loss += F.nll_loss(output, y_, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1] 
      #val_accuracy += pred.eq(y_.data.view_as(pred)).cpu().float().sum()
      val_accuracy += pred.eq(y_.data.view_as(pred)).float().sum()

  # Horovod: use test_sampler to determine the number of examples in
  # this worker's partition.
  val_loss /= len(valid_sampler)
  val_accuracy /= len(valid_sampler)

  # Horovod: average metric values across workers.
  val_loss = metric_average(val_loss, 'avg_loss')
  val_accuracy = metric_average(val_accuracy, 'avg_accuracy')

  # Horovod: print output only on first rank.
  if hvd.rank() == 0:
      print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
          val_loss, 100. * val_accuracy))

  

  #print("Accuracy of Test Data: {}".format(100*correct/total))
'''
