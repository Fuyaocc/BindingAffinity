import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from   torch.nn  import KLDivLoss
import torch.nn.functional as F

def js_div(p_output, q_output, get_softmax=True):
    kl_div = KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output,dim=-1)
        q_output = F.softmax(q_output,dim=-1)
    log_mean_output = F.log_softmax((p_output + q_output)/2, dim=-1)
    return (kl_div(log_mean_output, p_output) + kl_div(log_mean_output, q_output))/2

def mpnn_train(model,dataloader,optimizer,criterion,args):
    model.train()
    prelist = []
    truelist = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        data.to(args.device)
        pre =  model(data,True)
        label = data.y.unsqueeze(-1)
        loss = criterion(pre,label)
        prelist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return model,prelist,truelist,epoch_loss

def gcn_predict(model,dataloader,criterion,args,i,epoch):
    model.eval()
    epoch_loss=0
    prelist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        data = data.to(args.device)
        label = data.y.unsqueeze(-1)
        pre=model(data,False)
        loss = criterion(pre,label)
        prelist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        names += data.name
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return names,prelist,truelist,epoch_loss

def cnn_train(model,dataloader,optimizer,criterion,args):
    model.train()
    prelist = []
    truelist = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        x,y = data[0],data[1]
        x.to(args.device)
        pre =  model(x)
        label = y.unsqueeze(-1).to(torch.float32)
        loss = criterion(pre,label)
        prelist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return model,prelist,truelist,epoch_loss

def cnn_predict(model,dataloader,criterion,args,i,epoch):
    model.eval()
    epoch_loss=0
    prelist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        x,y,name = data[0],data[1],data[2]
        x.to(args.device)
        label = y.unsqueeze(-1)
        pre = model(x)
        loss = criterion(pre,label)
        prelist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        names += name
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return names,prelist,truelist,epoch_loss
