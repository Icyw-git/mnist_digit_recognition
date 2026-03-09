import pandas as pd #导包
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm


#步骤：
#数据预处理
#搭建模型
#训练模型
#测试模型
#模型调优



#模型定义
#CNN模型结构：输入层-卷积层1-批归一化1-激活函数1-池化层1-卷积层2-批归一化2-激活函数2-池化层2-全连接层1-激活函数3-dropout1-全连接层2-激活函数4-dropout2-全连接层3-输出层
#搭建模型
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1,padding=1) #特征图大小28
        self.bn1=nn.BatchNorm2d(6)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2) #特征图大小14
        self.cnn2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1) #特征图大小12
        self.bn2=nn.BatchNorm2d(16)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=1) #特征图大小11
        self.fc1=nn.Linear(1936,120)

        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,32)
        self.fc4=nn.Linear(32,10)
    def forward(self,x):
        x=self.cnn1(x)
        x=self.bn1(x)
        x=torch.relu(x)
        x=self.pool1(x)
        x=self.cnn2(x)
        x=self.bn2(x)
        x=torch.relu(x)
        x=self.pool2(x)
        x=x.reshape(x.shape[0],-1) #将特征图展平,第一个维度表示样本数量,第二个维度表示特征图的高度和宽度（由于池化层减少了空间维度,所以高度和宽度会减半）
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=torch.relu(x)
        x=self.fc3(x)
        x=torch.tanh(x)
        x=self.fc4(x)
        return x




def train_cnn():



    # 改进方式二：数据增强
    # 数据增强：通过对训练数据进行随机变换（如旋转、平移、缩放、裁剪等），可以增加训练数据的多样性，从而提高模型的泛化能力。
    transform_train=transforms.Compose([
        # transforms.RandomRotation(10), #随机旋转10度
        # transforms.RandomHorizontalFlip(), #随机水平翻转
        transforms.ToTensor(), #将图像转换为张量
        transforms.Normalize(mean=[0.1307],std=[0.3081]) #归一化,将像素值从[0,1]映射到[-1,1]

    ])
    transform_test=transforms.Compose([
        transforms.ToTensor(), #将图像转换为张量
        transforms.Normalize(mean=[0.1307],std=[0.3081]) #归一化,将像素值从[0,1]映射到[-1,1]
    ])



    train_dataset = MNIST(root='./dataset', train=True, download=True, transform=transform_train)
    test_dataset = MNIST(root='./dataset', train=False, download=True, transform=transform_test)  # 定义数据集



    #训练CNN模型
    train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    epochs=20
    model_cnn=cnn() #实例化模型


    optimizer=optim.AdamW(model_cnn.parameters(),lr=0.001,weight_decay=1e-2) #定义优化器
    # scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0.0001) #改进方式一：定义余弦退火学习率调度器，T_max为最大训练轮数，eta_min为最小学习率，在训练过程中进行学习率调整
    #lr_scheduler先进行学习率调整，之后optimizer进行参数更新，optimizer的学习率是scheduler调整后的学习率
    criterion=nn.CrossEntropyLoss() #定义损失函数

    #改进：选取训练过程中准确率最高的一组参数进行保存，而不是每轮都保存模型参数，节省存储空间，同时也能保证模型性能
    best_acc=0
    best_epoch=0



    for epoch in range(epochs): #每轮训练
        model_cnn.train() #转换为训练模式

        #使用tqdm库可视化训练进度，显示进度条
        pbar=tqdm(train_dataloader,desc=f'Epoch {epoch+1}/{epochs}')


        total_samples,total_loss,total_correct,acc=0,0,0,0#定义训练记录参数
        start=time.time()
        for batch_idx,(x,y) in enumerate(pbar):
            optimizer.zero_grad() #梯度清零
            y_pred=model_cnn(x) #模型预测
            loss=criterion(y_pred,y) #计算误差
            loss.backward() #反向传播
            optimizer.step() #更新参数
            # scheduler.step() #更新学习率
            total_samples+=x.shape[0]
            total_loss+=loss.item()*x.shape[0]
            total_correct+=(torch.argmax(y_pred,dim=1)==y).sum()
            acc=total_correct/total_samples
            end=time.time()

            pbar.set_postfix(loss=loss.item(),acc=acc) #更新进度条显示当前批次的损失和准确率



        print(f'epoch:{epoch},loss:{total_loss/total_samples},acc:{acc},time:{end-start}')

    #保存模型
    torch.save(model_cnn.state_dict(),'./models/cnn.pth') #保存模型



def train_knn():
    train_dataset = MNIST(root='./dataset', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='./dataset', train=False, download=True, transform=ToTensor())  # 定义数据集

    estimator = KNeighborsClassifier()
    param_grids = {
        'n_neighbors': [3, 5, 9],
        # 'metric': ['euclidean','minkowski']

    }
    gridsearch = GridSearchCV(estimator, param_grid=param_grids, cv=5)


    #数据准备
    x_train=train_dataset.data.numpy() #转换为numpy格式
    x_train=x_train.reshape(-1,28*28) #展平成一维
    x_test=train_dataset.targets.numpy() #提取标签列
    y_train=test_dataset.data.numpy()
    y_train=y_train.reshape(-1,28*28)
    y_test=test_dataset.targets.numpy()
    print(y_test.shape)
    print(y_train.shape)

    transfer=StandardScaler() #数据标准化
    x_train=transfer.fit_transform(x_train)
    y_train=transfer.fit_transform(y_train)



    gridsearch.fit(x_train,x_test) #训练模型
    print(gridsearch.best_params_) #输出最佳参数，写在fit之后，才能得到最佳参数

    joblib.load(gridsearch,'./models/knn')

    #测试
    y_pred=gridsearch.estimator(y_train)
    acc=accuracy_score(y_test,y_pred)
    print(acc)


def train_xgboost():
    #数据预处理
    train_dataset=MNIST(root='./dataset', train=True, download=True)
    test_dataset=MNIST(root='./dataset', train=False, download=True)
    x_train=train_dataset.data
    x_train=x_train.reshape(-1,28*28)
    y_train=train_dataset.targets
    x_test=test_dataset.data
    x_test=x_test.reshape(-1,28*28)
    y_test=test_dataset.targets

    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #构建模型
    xgboost=xgb.XGBClassifier()
    # params={
    #     'max_depth':[3,5,7],
    #     'n_estimators':[100,120,150],
    #     'learning_rate':[0.01,0.1],
    #
    # }
    # gridsearch=GridSearchCV(xgboost,param_grid=params,cv=5)
    # print(gridsearch.best_params_)
    xgboost.fit(x_train,y_train)
    y_pred=xgboost.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    print(f'Accuracy:{acc}')

    #保存模型
    joblib.dump(xgboost,'./models/xgboost.joblib')





    









#改进方式一：调整学习率 使用学习率调度器或余弦退火
#学习率调度器：是一种在训练过程中动态调整学习率的技术，它可以帮助模型在训练初期快速收敛，同时在训练后期避免过拟合。
#余弦退火：是一种学习率调度器，它根据训练轮数动态调整学习率，使得学习率在训练初期快速下降，在训练后期缓慢下降。
#这里尝试余弦退火,测试发现首轮准确率提升至95%
#方式二：
# 数据增强：通过对训练数据进行随机变换（如旋转、平移、缩放、裁剪等），可以增加训练数据的多样性，从而提高模型的泛化能力。
















if '__main__'==__name__:
    model=cnn()
    summary(model,(1,28,28))
    train_cnn()
    # train_xgboost()

    # train_cnn() #测试发现首轮准确率较低，为90%左右，进而思考调参方式来提高准确率



