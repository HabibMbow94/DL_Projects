from importlib.metadata import metadata
from typing_extensions import Required
from xml.parsers.expat import model
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch
import train
from models import resnet, CNN


from matplotlib import transforms
from torch import batch_norm
from data import dataloader
from config import args
from utils import plot, transform
metadata = args.metadata
path = args.path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data = dataloader.CustomDataSet(metadata, path, transform= transform)

batch_size = args.bs
train_size = args.train_size
train_size= int(train_size*len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#Model 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m',  '--model_name', help= 'This is the name of the model', required= True)
parser.add_argument('-n', '--num_epochs', help= 'This is the number of epochs', type=int, required = True)

mains_args = vars(parser.parse_args())
num_epochs = mains_args['num_epochs']

if mains_args['model_name'].lower()=='resnet':
    model = resnet()


if mains_args['model_name'].lower()=='cnn':
    model = CNN()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=args.wd)

modlel_trained, percent, val_loss, val_acc, train_loss, train_acc = train.train(model, criterion, train_loader, val_loader, optimizer, num_epochs)

plot(train_loss, val_loss)