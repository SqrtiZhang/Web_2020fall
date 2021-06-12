import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from focal_loss import FocalLoss
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import BertPreTrainedModel


from transformers import BertModel

from data_process import load_train, map_id_rel
 
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

rel2id, id2rel = map_id_rel()

#print(len(rel2id))
#print(id2rel)



USE_CUDA = torch.cuda.is_available()
#USE_CUDA=False

data=load_train(r"../data/train.json")
train_text=data['text']
train_mask=data['mask']
train_label=data['label']

train_text = [ t.numpy() for t in train_text]
train_mask = [ t.numpy() for t in train_mask]

train_text=torch.tensor(train_text)
train_mask=torch.tensor(train_mask)
#print(train_text)
train_label=torch.tensor(train_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

data=load_train(r"../data/dev1.json")
dev_text=data['text']
dev_mask=data['mask']
dev_label=data['label']

dev_text = [ t.numpy() for t in dev_text]
dev_mask = [ t.numpy() for t in dev_mask]

dev_text=torch.tensor(dev_text)
dev_mask=torch.tensor(dev_mask)
dev_label=torch.tensor(dev_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

print("--eval data--")
print(dev_text.shape)
print(dev_mask.shape)
print(dev_label.shape)

# exit()
#USE_CUDA=False

if USE_CUDA:
    print("using GPU")

train_dataset = torch.utils.data.TensorDataset(train_text,train_mask,train_label)
dev_dataset = torch.utils.data.TensorDataset(dev_text,dev_mask,dev_label)

def get_train_args():
    labels_num=len(rel2id)
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--nepoch',type=int,default=5)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--gpu',type=bool,default=True)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--num_labels',type=int,default=labels_num)
    parser.add_argument('--data_path',type=str,default='.')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    model = BertForSequenceClassification.from_pretrained(r"../../../data/Sqrti/bert-uncase/bert-base-uncased",num_labels=opt.num_labels)
    return model


def eval(net, dataset, batch_size):
    net.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y=y.cuda()

            outputs= net(text, attention_mask=mask,labels=y)
            #print(y)
            loss, logits = outputs[0],outputs[1]
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
            #print(y.data)
            #print(predicted.data)
            #print(predicted.data.eq(y.data))
            s = ("Acc:%.3f" %((1.0*correct.numpy())/total))
        acc= (1.0*correct.numpy())/total
        print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
        return acc


def train(net,dataset,num_epochs, learning_rate,  batch_size):
    net.train()
    #print(net.parameters(()))
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 20, 50, 80], gamma=0.5, last_epoch=-1)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate )
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre=0

    for epoch in range(num_epochs):
        #scheduler.step()
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            optimizer.zero_grad()
            #print(type(y))
            #print(y)
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y = y.cuda()
            #print(text.shape)
            #print(type(net(text, attention_mask=mask,labels=y)))
            #print(y)
            #print(mask)
            #print(text)
            output = net(text, attention_mask=mask,labels=y)
            loss = output['loss']
            logits = output['logits']
            #print(loss.shape)
            #print(loss)
            #print("predicted",predicted)
            #print("answer", y)
            #log = logits.cuda()
            #log = logits
            #todo: norm weight
            '''loss_fn = FocalLoss(weight = torch.Tensor([1, 1.39824732, 1.3019039,  1.36501901 ,1.90703851, 1.79724656,
 1.67953216 ,1.82002535 ,2.36963696 ,2.65925926]).cuda())
            loss_fn  = FocalLoss()
            loss = loss_fn(log, y)'''
            loss.backward()
            #print(loss)
            optimizer.step()
            #loss.mean().backward()
            #optimizer.step()
            #print(outputs[1].shape)
            #print(output)
            #print(outputs[1])
            _, predicted = torch.max(logits.data, 1)


            #print(loss)

            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss=loss.detach().cpu()
        print("epoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.cpu().numpy().tolist(), "total", total, "Acc:", correct.cpu().numpy().tolist()/total)
        acc = eval(model, dev_dataset, 32)
        if acc > pre:
            pre = acc
            torch.save(model, "../model/" + str(acc)+'.pth')
    return

opt = get_train_args()
model=get_model(opt)
#model = torch.nn.DataParallel(model)
#model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    model=model.cuda()

#eval(model,dev_dataset,8)

train(model,train_dataset,100,0.001,32)
#eval(model,dev_dataset,8)

