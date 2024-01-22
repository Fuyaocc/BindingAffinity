import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from   torch.nn  import KLDivLoss
import torch.nn.functional as F

def mpnn_train(model,dataloader,optimizer,criterion, args,epoch_idx):
    model.train()
    predlist = []
    truelist = []
    names = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        y_h = data.y.unsqueeze(-1)
        y =  model(data)
        loss = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += data.y.cpu().tolist()
        names += data.name
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return model,names,predlist,truelist,epoch_loss

def gcn_predict(model,dataloader,criterion,args):
    model.eval()
    epoch_loss = 0.0
    predlist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        y =  model(data)
        y_h = data.y.unsqueeze(-1)
        mse = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += y_h.squeeze(-1).cpu().tolist()
        names += data.name
        epoch_loss += mse.detach().item()
    epoch_loss /= (batch_id+1)
    return names,predlist,truelist,epoch_loss

def cnn_train(model,dataloader,optimizer,criterion,args):
    model.train()
    predlist = []
    truelist = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        x,y = data[0],data[1]
        x.to(args.device)
        y.to(args.device)
        pre =  model(x)
        label = y.unsqueeze(-1).to(torch.float32).to(args.device)
        loss = criterion(pre,label)
        predlist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return model,predlist,truelist,epoch_loss

def cnn_predict(model,dataloader,criterion,args,i,epoch):
    model.eval()
    epoch_loss=0
    predlist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        x,y,name = data[0],data[1],data[2]
        x.to(args.device)
        y.to(args.device)
        label = y.unsqueeze(-1).to(torch.float32).to(args.device)
        pre = model(x)
        loss = criterion(pre,label)
        predlist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        names += name
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return names,predlist,truelist,epoch_loss
