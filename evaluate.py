import torch
import joblib
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import accuracy_score


from model import cnn

test_dataset=MNIST('./dataset',train=False,transform=transforms.ToTensor())

def evaluate():
    model_cnn=cnn()
    model_cnn.load_state_dict(torch.load('./models/cnn.pth')) #加载保存模型的参数

    model_cnn.eval() #将模型设置为评估模式
    test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False)
    total_correct, total_samples = 0, 0
    with torch.no_grad(): #关闭梯度计算，节省内存，验证时需要

        for x,y in test_dataloader:
            y_pred_cnn=model_cnn(x)
            total_samples+=x.shape[0]
            total_correct+=(torch.argmax(y_pred_cnn, dim=1) == y).sum() 
    acc=total_correct/total_samples
    print('Accuracy_cnn:',acc.item())


    #评价xgboost
    model_xgb=joblib.load('./models/xgboost.joblib')
    x_test=test_dataset.data.numpy() #从张量转为numpy格式
    x_test=x_test.reshape(-1,28*28) #展平成一维
    y_test=test_dataset.targets.numpy() #targets表示标签列，转为numpy格式
    y_pred_xgb=model_xgb.predict(x_test)
    acc=accuracy_score(y_test,y_pred_xgb)
    print('Accuracy_xgb:',acc)



if __name__=='__main__':
    evaluate()