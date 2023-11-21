import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from   torch.nn  import KLDivLoss
import torch.nn.functional as F
import torch.nn as nn

from  torchsummary import summary

def js_div(p_output, q_output, get_softmax=True):
    kl_div = KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output,dim=-1)
        q_output = F.softmax(q_output,dim=-1)
    log_mean_output = F.log_softmax((p_output + q_output)/2, dim=-1)
    return (kl_div(log_mean_output, p_output) + kl_div(log_mean_output, q_output))/2

def run_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,num_layers):
    model.train()
    model.to(device)
    dataitr=iter(dataloader)
    f=open(f'./tmp/train/val{i}/train_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    epoch_loss=0.0
    for batch_id,data in enumerate(dataitr):
        optimizer.zero_grad()
        features=data[:-1]
        for feature in range(len(features)):
            features[feature]=features[feature].to(device)
        label = data[-1].to(device)
        pre = model(features[0],features[1],"train")
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32)
        loss = criterion(pre, label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i]))
            f.write(str(float(label[i])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss

def gcn_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,epsilon,alpha):
    model.train()
    model.to(device)
    f=open(f'./tmp/train/val{i}/train_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    epoch_loss=0.0
    epoch_normal_mse=0.0
    epoch_against_mse=0.0
    epoch_against_js=0.0
    for batch_id,data in enumerate(dataloader):
        optimizer.zero_grad()
        label = data.y
        names=data.name
        pre = model(data,False,device)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
        total_loss = criterion(pre,label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
            f.write(names[i])
            f.write('\t\t')
            f.write(str(float(label[i][0])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        total_loss.backward()
        optimizer.step()
        epoch_loss += (total_loss.detach().item())
    epoch_loss /= (batch_id+1)
    return model,prelist,truelist,epoch_loss

def mpnn_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,epsilon,alpha,kl_loss):
    model.train()
    prelist = []
    truelist = []
    epoch_loss=0.0
    for batch_id,data in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        label = data.y
        pre=model(data,True,device)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
        loss = criterion(pre,label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
        #     f.write(names[i])
        #     f.write('\t\t')
        #     f.write(str(float(label[i][0])))
        #     f.write('\t\t')
        #     f.write(str(float(pre[i][0])))
        #     f.write('\n')
        loss.backward()
        optimizer.step()

        epoch_loss += (loss.detach().item())
        
    epoch_loss /= (batch_id+1)
    
    return model,prelist,truelist,epoch_loss

def run_predict(model,dataloader,criterion,device,i,epoch,num_layers):
    model.eval()
    epoch_loss=0
    dataitr=iter(dataloader)
    prelist = []
    truelist = []
    for batch_id,data in enumerate(dataitr):
        features=data[:-1]
        for feature in range(len(features)):
            features[feature]=features[feature].to(device)
        label = data[-1].to(device)
        pre = model(features[0],features[1],num_layers,False)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32)
        loss = criterion(pre, label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
        #     f.write(str(float(label[i])))
        #     f.write('\t\t')
        #     f.write(str(float(pre[i][0])))
        #     f.write('\n')
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss


def gcn_predict(model,dataloader,criterion,device,i,epoch):
    model.eval()
    epoch_loss=0
    # f=open(f'./tmp/test/val{i}/test_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        data = data.to(device)
        label = data.y        
        pre=model(data,False,device)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
        loss = criterion(pre,label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
            names.append(data.name[i])
            # f.write(names[i])
            # f.write('\t\t')
            # f.write(str(float(label[i][0])))
            # f.write('\t\t')
            # f.write(str(float(pre[i][0])))
            # f.write('\n')
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return names,prelist,truelist,epoch_loss
