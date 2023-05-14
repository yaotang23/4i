import pandas as pd
from torch.utils import data
import os
from sklearn.model_selection import KFold
from torchvision import transforms as T 
from PIL import Image
import random
import torch
from tqdm import tqdm
import sys
import numpy as np
from scipy.special import lambertw
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC, CohenKappa, MetricCollection

def check_id(id):
    if id>=0:
        return 0
    return 1

def get_ori_id(id):
    if id<0:
        return -id-1
    return id

def get_gamma(p=0.3):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

class OCTA_DATA(data.Dataset):
    def __init__(self,dataset_path, csv_path=None,type=0,val_prob=0.3,k_fold_n_splits=5):
        print("Starting to load data for dataset_path:",dataset_path)
        self.type=type
        self.y=[]
        self.x=[]
        self.dataset_path=dataset_path
        
        self.da={
            'brightness': 0.4,  # how much to jitter brightness # 0.8,1.2
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from # vit:-180
            'img_size': 244
         }
        self.transform = T.Compose([
                T.Resize((640,640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(
                    size=((self.da['img_size'], self.da['img_size'])),
                    scale=self.da['scale'],
                    ratio=self.da['ratio']
                ),

                T.ColorJitter(
                    brightness=self.da['brightness'],
                    contrast=self.da['contrast'],
                ),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
          
            ])
    
        if type==0:
            tem_y=pd.read_csv(csv_path)
            tem_name=[]
            tem_type=[]
            for _,i_row in tem_y.iterrows():
                img_name=i_row['image name']
                img_label=get_ori_id(i_row['DR grade'])
                tem_name.append(img_name)
                if random.random()<val_prob:
                    tem_type.append(-img_label-1)
                else:
                    tem_type.append(img_label)
            dataframe = pd.DataFrame({'image name':tem_name,'DR grade':tem_type})
            dataframe.to_csv(csv_path,index=False,sep=',')

        if type==0 or type==1: #train or val
            img_list = [[] for _ in range(3)]
            tem_y=pd.read_csv(csv_path)
            for _,i_row in tem_y.iterrows():
                img_name=i_row['image name']
                img_label=i_row['DR grade']
                if type==check_id(img_label):
                    img_list[get_ori_id(img_label)].append(img_name)
            print(len(img_list))
            for class_id in range(3):
                for i in range(len(img_list[class_id])):
                    self.x.append(img_list[class_id][i])
                    self.y.append(class_id)
        else:
            path_list=os.listdir(dataset_path)
            path_list.sort(key=lambda x:int(x[:-4])) #将'.jpg'左边的字符转换成整数型进行排序
            for image_id in path_list:
                self.x.append(image_id)
           
    def __getitem__(self, key):
        if self.type==0 or self.type==1:
            path=self.dataset_path+self.x[key]
            data=Image.open(path).convert('RGB')
            data=self.transform(data)
            label=self.y[key]
            return data, label, self.x[key]
        else:
            path=self.dataset_path+self.x[key]
            data=Image.open(path).convert('RGB')
            data=self.transform(data)
            label=random.randint(0,2)
            return data, label, self.x[key]
    def __len__(self):
        return len(self.x)
    
import torch
from torch.nn import functional as F

def quadratic_kappa_coefficient(output, target):
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK

def quadratic_kappa_loss(output, target, scale=2.0):
    QWK = quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss

class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0, n_classes=3):
        super().__init__()
        self.scale = scale
        self.n_classes = n_classes

    def forward(self, output, target):
        # Keep trace of output dtype for half precision training
        target = F.one_hot(target.squeeze(), num_classes=self.n_classes).to(target.device).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return quadratic_kappa_loss(output, target, self.scale)    

ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=3.0, size_average=False, device='cuda'):
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

from pytorch_toolbelt.inference import tta
    
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = QWKLoss2() # 或者FocalLossAdaptive()
    accu_loss = torch.zeros(1).to(device)  # 用于累计损失
    accu_num = torch.zeros(1).to(device)   # 用于累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):  # step对应索引 data对应所传入的dataloader参数中的每个元素
        images, labels,_ = data
        sample_num += images.shape[0]  # batchsize维度的值求和，即样本数量
        #前向
        pred = model(images.to(device))  # 预测结果
        pred_classes = torch.max(pred, dim=1)[1]
        # 在dim=1维度找到预测值最大的值，即为预测类；
        # torch.max()得到{max, max_indices}，使用torch[1]取出该张量的dim=1维度的向量，即max_indices向量（size=batchsize*类别数），为预测最大值对应的类别索引
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 判断max_indices与label（label是batchsize*类别数的向量）是否相等，相等返回True；并累计预测正确的样本数
        
        # TTA
        # inputs=images
        # alpha=1
        # lam = np.random.beta(alpha,alpha)
        # index = torch.randperm(images.size(0)).cuda()
        # inputs = lam*images + (1-lam)*images[index,:]
        # targets_a, targets_b = labels, labels[index]
        # outputs = model(inputs.to(device))
        # loss = lam * loss_function(outputs, targets_a.to(device)) + (1 - lam) * loss_function(outputs, targets_b.to(device))
        
        # Truly functional TTA for image classification using horizontal flips:
        # outputs = tta.fliplr_image2label(model, inputs)

        # Truly functional TTA for image segmentation using D4 augmentation:
        # outputs = tta.d4_image2mask(model, inputs)

        # TTA using wrapper module:
        # tta_model = tta.TTAWrapper(model, tta.fivecrop_image2label, crop_size=512)
        # outputs = tta_model(inputs)

        loss = loss_function(outputs, labels.to(device))  # loss函数的输入参数是[pre，label]
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):  # loss = inf or -inf or nan 的时候结束训练
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels,_ = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

if __name__=='__main__':
    dataset_path='/root/autodl-tmp/4i/Dataset/C. Diabetic Retinopathy Grading'

    test_path=dataset_path+'/1. Original Images/b. Testing Set/'
    train_path=dataset_path+'/1. Original Images/a. Training Set/'
    ground_path=dataset_path+'/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'

    test_dataset=OCTA_DATA(test_path,None,2)
    train_dataset=OCTA_DATA(train_path,ground_path,0)
    val_dataset=OCTA_DATA(train_path,ground_path,1)
    img,label,name=train_dataset.__getitem__(1)
    print(label, name)