from  torch.utils.data import Dataset

def pickfold(featurelist, labelList,namelist, train_index, test_index):
    x_train=[]
    y_train=[]
    name_train=[]
    x_test=[]
    y_test=[]
    name_test=[]
    for index in train_index:
        x_train.append(featurelist[index])
        y_train.append(labelList[index])
        name_train.append(namelist[index])
    for index in test_index:
        x_test.append(featurelist[index])
        y_test.append(labelList[index])
        name_test.append(namelist[index])
    return  x_train,y_train,name_train,x_test,y_test,name_test

class MyDataset(Dataset):
    def __init__(self, featurelist, labellsit, namelist):
        self.featurelist = featurelist
        self.labellist = labellsit
        self.namelist = namelist
    
    def __getitem__(self,index):
        return  [self.featurelist[index], self.labellist[index],self.namelist[index]]

    def __len__(self): 
        return len(self.featurelist)
    

def gcn_pickfold(featurelist,train_index, test_index):
    train=[]
    test=[]
    for index in train_index:
        train.append(featurelist[index])
    for index in test_index:
        test.append(featurelist[index])
    return  train,test
    
class MyGCNDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]