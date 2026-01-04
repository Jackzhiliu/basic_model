import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# print("Inside script—cuda.is_available():", torch.cuda.is_available())
# print("Inside script—device_count():   ", torch.cuda.device_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("→ running on", device)

# import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as f
import math as m
from helper1 import inverse_layerwise_training, conv_train_2_fc_layer_last,inc_train_2_layer_e2e_acce,conv_train_2_fc_layer_last_e2e, DEVICE_
import my_module1 as mm
from my_module1 import m0,m1,m2,m3,m4,m5,m6,m7,m8,m9
import my_functional as mf
# import scipy.io as sio
import math
import sklearn.model_selection
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

import time
# import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm #从tqdm库中导入tadm类

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 64
FOLDER_e2e=mm.FOLDER_e2e
if not os.path.exists(FOLDER_e2e):
    os.mkdir(FOLDER_e2e)
# define transforms
# transforms = transforms.Compose([transforms.ToTensor()])  #
# transforms = None

# download and create datasets
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
def get_set_weight(model1,model2,index):
    for i in range(len(index)):
        w=model1.layers[index[i]].weights.weight.data
        model2.layers[index[i]].weights.weight.data=w
        # print (i,index[i])
def save_best(modelnum,j,acc_lst,bestaccu,model):
    if acc_lst[1] > bestaccu:
        name = (FOLDER_e2e + '/model' + '%d' % modelnum + '_mnistbest.pkl')
        torch.save(model, name)
        bestaccu = acc_lst[1]
        model.save_best_accu(modelnum, j, acc_lst)
        print(j, acc_lst[1])
    name0 = (FOLDER_e2e + '/model' + '%d' % modelnum + '_mnistlast.pkl')
    torch.save(model, name0)
    return bestaccu

#########train-valid-split#######################
class GrayscaleToRGB:
    def __call__(self, img):
        return img.convert("RGB")
# train_set =datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
# GrayscaleToRGB(),
#             transforms.Resize([32,32]),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(32, 4),
#
#     transforms.ToTensor(),
#         ]), download=True)
# print(len(train_set))
# val_set =datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
# GrayscaleToRGB(),
#             transforms.Resize([32,32]),
#             # transforms.RandomHorizontalFlip(),
#             # transforms.RandomCrop(32, 4),
#
#             transforms.ToTensor(),
#         ]), download=True)

train_set = datasets.SVHN(root='./data', split='train', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        # normalize,
    ]), download=True)
val_dataset = datasets.SVHN(root='./data', split='train', transform=transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    # transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    # normalize,
]), download=True)
# train_set = datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
#     transforms.Resize(size=(32, 32)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, 4),
#     # transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     # normalize,
# ]), download=True)
# print(len(train_set))
# val_set = datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomCrop(32, 4),
#     transforms.Resize(size=(32, 32)),
#     transforms.ToTensor(),
#     # normalize,
# ]), download=True)
# train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     # transforms.Resize(size=(224, 224)),
#     transforms.RandomCrop(32, 4),
#     transforms.ToTensor(),
#     # normalize,
# ]), download=True)
# print(len(train_set))
# val_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomCrop(32, 4),
#     # transforms.RandomCrop(224, 4),
#     # transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     # normalize,
# ]), download=True)

# num_train = len(train_set)
# print(num_train)
# indices = list(range(num_train))
# valid_size = 0.1
# split = int(np.floor(valid_size * num_train))
# # print('split',num_train-split)
# np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# train_dataset = torch.utils.data.Subset(train_set, train_indices)
# val_dataset = torch.utils.data.Subset(val_set, val_indices)
torch.manual_seed(42)
train_size = int(0.7 * len(train_set))
val_size = len(train_set) - train_size

# Split the dataset using the fixed seed
train_dataset, train_onlineset = random_split(train_set, [train_size, val_size])
train_loaderA = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderA = DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
print('len',len(train_dataset),len(val_dataset))

# shuffle the sequences of samples before slicing into batches
train_loaderB = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderB= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderC = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderC= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderD = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderD= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderE = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderE= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderF = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderF= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderG = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderG= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderH = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderH= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderI = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderI= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)
train_loaderJ = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
val_loaderJ= DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False)


# test_loader = torch.utils.data.DataLoader(
#         datasets.KMNIST(root='./data', train=False, transform=transforms.Compose([
#             transforms.Resize([32,32]),
#             transforms.ToTensor(),
#         ])),
#         batch_size=BATCH_SIZE, shuffle=False,
#         num_workers=4, pin_memory=True)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ running on", DEVICE)
# DEVICE='cuda'
N_CLASSES = 10

print(DEVICE)
mm.DEVICE = DEVICE
DEVICE_[0] = DEVICE

# model has more and more layers
model0=m0(10,train_loaderA)
model1=m1(10,train_loaderB)
model2=m2(10,train_loaderC)
model3=m3(10,train_loaderD)
model4=m4(10,train_loaderE)
model5 =m5(10,train_loaderF)
model6 =m6(10,train_loaderG)
model7 =m7(10,train_loaderH)
model8 = m8(10,train_loaderI)
model9 = m9(10,train_loaderJ)
# model0.load_weights('./saved_models_KMNIST/_1_two_layer_199')
# model1.load_weights('./saved_models_KMNIST/_2_two_layer_199')
# model2.load_weights('./saved_models_KMNIST/_3_two_layer_199')
# model3.load_weights('./saved_models_KMNIST/_4_two_layer_199')
# acc_lst0 = model0.evaluate_both(1, train_loaderA, test_loader)
# acc_lst1 = model1.evaluate_both(1, train_loaderA, test_loader)
# acc_lst2 = model2.evaluate_both(1, train_loaderA, test_loader)
# acc_lst3 = model3.evaluate_both(3, train_loaderD, test_loader)
# print('acc_lst0',acc_lst0)
# print('acc_lst1',acc_lst1)
# print('acc_lst2',acc_lst2)
# print('acc_lst3',acc_lst3)
#
# model0=torch.load('./saved_models_KMNIST5e-3all/model0_kmnistlast.pkl')
# model1=torch.load('./saved_models_KMNIST5e-3all/model1_kmnistlast.pkl')
# model2 =torch.load('./saved_models_KMNIST5e-3all/model2_kmnistlast.pkl')
# model3 =torch.load('./saved_models_KMNIST5e-3all/model3_kmnistlast.pkl')
# model4 =torch.load('./saved_models_KMNIST5e-3all/model4_kmnistlast.pkl')
# model5 =torch.load('./saved_models_KMNIST5e-3all/model5_kmnistlast.pkl')
# model6 =torch.load('./saved_models_KMNIST5e-3all/model6_kmnistlast.pkl')
# model7 =torch.load('./saved_models_KMNIST5e-3all/model7_kmnistlast.pkl')
# model8 =torch.load('./saved_models_KMNIST5e-3all/model8_kmnistlast.pkl')
# model9 =torch.load('./saved_models_KMNIST5e-3all/model9_kmnistlast.pkl')

# acc_lst4 = model4.evaluate_both(4, train_loaderE, test_loader)
# acc_lst5 = model5.evaluate_both(5, train_loaderF, test_loader)
# acc_lst6 = model6.evaluate_both(6, train_loaderG, test_loader)
# acc_lst7 = model7.evaluate_both(7, train_loaderH, test_loader)
# acc_lst8 = model8.evaluate_both(8, train_loaderI, test_loader)
# acc_lst9 = model9.evaluate_both(9, train_loaderJ, test_loader)


model0 = model0.float();
model1 = model1.float();model2 = model2.float();model3 = model3.float();
# model4 = model4.float();
# model5 = model5.float();model6 = model6.float();model7 = model7.float();model8 = model8.float();model9 = model9.float();


t0 = time.time()
bestaccu0 =0;bestaccu1 =0;bestaccu2 =0;bestaccu3 =0
bestaccu4=0;bestaccu5 =0;bestaccu6 =0;bestaccu7 =0;bestaccu8 =0;bestaccu9 =0;


# acc_lst0=[0,0];acc_lst1=[0,0];acc_lst2=[0,0];acc_lst3=[0,0];acc_lst4=[0,0];acc_lst5=[0,0];acc_lst6=[0,0];acc_lst7=[0,0];
print('************** Training last 2 layers ****************')
# acc_lst0 = model0.evaluate_both(0, test_loader, test_loader)
# acc_lst1 = model1.evaluate_both(1, test_loader, test_loader)
# acc_lst2 = model2.evaluate_both(2, test_loader, test_loader)
# acc_lst3 = model3.evaluate_both(3, test_loader, test_loader)
# acc_lst4 = model4.evaluate_both(4, test_loader, test_loader)
# acc_lst5 = model5.evaluate_both(5, test_loader, test_loader)
# acc_lst6 = model6.evaluate_both(6, test_loader, test_loader)
# acc_lst7 = model7.evaluate_both(7, test_loader, test_loader)
# acc_lst8 = model8.evaluate_both(8, test_loader, test_loader)
# acc_lst9 = model9.evaluate_both(9, test_loader, test_loader)
# print('acc_lst0',acc_lst0)
# print('acc_lst1',acc_lst1)
# print('acc_lst2',acc_lst2)
# print('acc_lst3',acc_lst3)
# print('acc_lst4',acc_lst4)
# print('acc_lst5',acc_lst5)
# print('acc_lst6',acc_lst6)
# print('acc_lst7',acc_lst7)
# print('acc_lst8',acc_lst8)
# print('acc_lst9',acc_lst9)
dataiterA = iter(train_loaderA)
dataiterB = iter(train_loaderB)
dataiterC = iter(train_loaderC)
dataiterD = iter(train_loaderD)
dataiterE = iter(train_loaderE)
dataiterF = iter(train_loaderF)
dataiterG = iter(train_loaderG)
dataiterH = iter(train_loaderH)
dataiterI = iter(train_loaderI)
dataiterJ = iter(train_loaderJ)
truefor=1
# get_set_weight(model3, model4, [0, 2, 4])
# get_set_weight(model0, model1, [0])
learnrate=0.001
t0 = time.time()

# number of epochs
for j in range(0,200):
      # if j >20:
      #     learnratete=0.001
      # elif j>10:
      #     learnrate = 0.0005
      # # print('epoch',j)

      # number of batches
      for i in range(0,len(train_loaderD)):
        # print(i)


        # if i % 10000==0:
        #      print('now is train process epoch ',j,' batch',i)
        # if i ==1:
        # if i < percentage*len(train_loader):



        try:
             # extract a batch of samples
             xA, yA = next(dataiterA)
             # # if i  == 1:
             # #    print(y_noaug)
             # # print('model0',i)
             # t1 = time.time()

             # train conv and fc weights
             inc_train_2_layer_e2e_acce(model0,i,j,xA,yA,ker=2,stri=2,epochs=200, gain=learnrate, pool_layer='max',true_for=truefor)
             # pass the conv weights to new model
             get_set_weight(model0,model1,[0])
             # print('model1', i)

             # xB, yB = next(dataiterB)
             inc_train_2_layer_e2e_acce(model1, i, j, xA, yA, ker=2,stri=2,epochs=200, gain=learnrate,  pool_layer='max',true_for=truefor)
             get_set_weight(model1,model2,[0,2])
             # print('model2', i)
             # xC, yC = dataiterC.next()
             inc_train_2_layer_e2e_acce(model2,  i,j, xA, yA, ker=2,stri=2,epochs=200, gain=learnrate, pool_layer=False,true_for=truefor)
             get_set_weight(model2,model3,[0,2,4])
             # print('model3', i)
             # xD, yD = dataiterD.next()
             inc_train_2_layer_e2e_acce(model3,i, j, xA, yA,ker=2,stri=2,  epochs=200, gain=learnrate, pool_layer='max',true_for=truefor)
             get_set_weight(model3, model4, [0, 2, 4,5])
             # print('model4', i)
             # xE, yE = dataiterE.next()
             inc_train_2_layer_e2e_acce(model4,  i,j, xA, yA, ker=2,stri=2, epochs=200, gain=learnrate,pool_layer=False, true_for=truefor)
             get_set_weight(model4, model5, [0, 2, 4,5,7])
             # print('model5', i)
             # xF, yF = dataiterF.next()

             inc_train_2_layer_e2e_acce(model5,  i,j, xA, yA,ker=2,stri=2,  epochs=200, gain=learnrate,pool_layer='max', true_for=truefor)
             get_set_weight(model5, model6, [0, 2, 4, 5,7,8])
             # print('model6', i)
             # xG, yG = dataiterG.next()
             inc_train_2_layer_e2e_acce(model6,  i,j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate, pool_layer=False, true_for=truefor)
             get_set_weight(model6, model7, [0, 2, 4, 5, 7,8,10])
             # print('model7', i)
             # xH, yH = dataiterH.next()
             inc_train_2_layer_e2e_acce(model7, i, j, xA, yA, ker=2,stri=2,epochs=200, gain=learnrate,pool_layer='max', true_for=truefor)
             # get_set_weight(model7, model8, [0, 2, 4, 5, 7,8,10,11])
             # # print('model8', i)
             # xI, yI = dataiterI.next()
             # conv_train_2_fc_layer_last_e2e(model8, i, j, xI, yI, epoch=200, loop=1, ran_mix=False, gain_=learnrate,
             #                                auto=True)
             #
             # get_set_weight(model8, model9, [0, 2, 4, 5, 7, 8, 10, 11, 14])
             # # print('model9', i)
             # xJ, yJ = dataiterJ.next()
             # conv_train_2_fc_layer_last_e2e(model9, i, j, xJ, yJ, epoch=200, loop=1, ran_mix=False, gain_=learnrate,
             #                                auto=True)

             # print('yB',yB);print('yC',yC);print('yD',yD);print('yE',yE);
             # get_set_weight(model8, model9, [0, 2, 4, 5, 7, 8, 10, 11])
            # if
            #  print('epoch',j,'batch',i,'time: ', time.time() - t1)
            
        # avoid running out of samples
        except StopIteration:
            print('StopIteration')
            dataiterA = iter(train_loaderA)
            # dataiterB = iter(train_loaderB)
            # dataiterC = iter(train_loaderC)
            # dataiterD = iter(train_loaderD)
            # dataiterE = iter(train_loaderE)
            # dataiterF = iter(train_loaderF)
            # dataiterG = iter(train_loaderG)
            # dataiterH = iter(train_loaderH)
            # dataiterI = iter(train_loaderI)
            # dataiterJ = iter(train_loaderJ)
            # xA, yA = dataiterA.next()
            # # if i  == 1:
            # # if i  == 1:
            # #    print(y_noaug)
            # # print('model0',i)
            # # t1 = time.time()
            inc_train_2_layer_e2e_acce(model0, i, j, xA, yA,ker=2,stri=2,epochs=200, gain=learnrate,  pool_layer='max', true_for=truefor)
            get_set_weight(model0, model1, [0])
            # print('model1', i)

            # xB, yB = dataiterB.next()
            inc_train_2_layer_e2e_acce(model1, i, j, xA, yA,ker=2,stri=2,epochs=200, gain=learnrate,  pool_layer='max', true_for=truefor)
            get_set_weight(model1, model2, [0, 2])
            # print('model2', i)
            # xC, yC = dataiterC.next()
            inc_train_2_layer_e2e_acce(model2, i, j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate, pool_layer=False, true_for=truefor)
            get_set_weight(model2, model3, [0, 2, 4])
            # print('model3', i)
            # xD, yD = dataiterD.next()
            inc_train_2_layer_e2e_acce(model3, i, j, xA, yA,ker=2,stri=2,epochs=200, gain=learnrate, pool_layer='max', true_for=truefor)
            get_set_weight(model3, model4, [0, 2, 4, 5])
            # print('model4', i)
            # xE, yE = dataiterE.next()
            inc_train_2_layer_e2e_acce(model4, i, j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate,pool_layer=False, true_for=truefor)
            get_set_weight(model4, model5, [0, 2, 4, 5, 7])
            # print('model5', i)
            # xF, yF = dataiterF.next()

            inc_train_2_layer_e2e_acce(model5, i, j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate,pool_layer='max', true_for=truefor)
            get_set_weight(model5, model6, [0, 2, 4, 5, 7, 8])
            # print('model6', i)
            # xG, yG = dataiterG.next()
            inc_train_2_layer_e2e_acce(model6, i, j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate,pool_layer=False, true_for=truefor)
            get_set_weight(model6, model7, [0, 2, 4, 5, 7, 8, 10])
            # print('model7', i)
            # xH, yH = dataiterH.next()
            inc_train_2_layer_e2e_acce(model7, i, j, xA, yA,ker=2,stri=2, epochs=200, gain=learnrate,pool_layer='max', true_for=truefor)

            # get_set_weight(model7, model8, [0, 2, 4, 5, 7, 8, 10, 11])
            # # print('model8', i)
            # xI, yI = dataiterI.next()
            # conv_train_2_fc_layer_last_e2e(model8, i, j, xI, yI, epoch=200, loop=1, ran_mix=False, gain_=learnrate,auto=True)
            #
            #
            # get_set_weight(model8, model9, [0, 2, 4, 5, 7, 8, 10, 11, 14])
            # # print('model9', i)
            # xJ, yJ = dataiterJ.next()
            # conv_train_2_fc_layer_last_e2e(model9, i, j, xJ, yJ, epoch=200, loop=1, ran_mix=False, gain_=learnrate,auto=True)
        # else:

      # acc_lst0 = model0.evaluate_both_e2e(0,percentage,train_loaderA, train_loaderA)#20 percent of train_loader for training
      print('time: ', time.time() - t0)
      if j % 1 == 0:
          acc_lst0 = model0.evaluate_both(1, train_loaderA, val_loaderA)
          acc_lst1 = model1.evaluate_both(1,train_loaderA, val_loaderA)
          acc_lst2 = model2.evaluate_both(2,train_loaderA, val_loaderA)
          acc_lst3 = model3.evaluate_both(3,train_loaderA, val_loaderA)
          acc_lst4 = model4.evaluate_both(4, train_loaderA, val_loaderA)
          acc_lst5 = model5.evaluate_both(5, train_loaderA, val_loaderA)
          acc_lst6 = model6.evaluate_both(6, train_loaderA, val_loaderA)
      #     # acc_lst7 = model7.evaluate_both(7, train_loaderH, val_loaderH)
      #     # acc_lst8 = model8.evaluate_both(8, train_loaderI, val_loaderI)
      #     # acc_lst9 = model9.evaluate_both(9, train_loaderJ, val_loaderJ)
      #     # acc_lst0 = model0.evaluate_both(1, test_loader, test_loader)
      #     # acc_lst1 = model1.evaluate_both(1, test_loader, test_loader)
      #     # acc_lst2 = model2.evaluate_both(2, test_loader, test_loader)
      #     # acc_lst3 = model3.evaluate_both(3, test_loader, test_loader)
      #     # acc_lst4 = model4.evaluate_both(4, test_loader, test_loader)
      #     # acc_lst5 = model5.evaluate_both(5, test_loader, test_loader)
      #     # acc_lst6 = model6.evaluate_both(6, test_loader, test_loader)
          acc_lst7 = model7.evaluate_both(7, train_loaderA, val_loaderA)
          # acc_lst8 = model8.evaluate_both(8, test_loader, test_loader)
          # acc_lst9 = model9.evaluate_both(9, test_loader, test_loader)
      #     # if j % 30 == 0:##test every 30 epochs:
      #     #     # acc_lst0 = model0.evaluate_both_e2e(0, percentage, train_loaderA,
      #     #     #                                     val_loader)  # 20 percent of train_loader for training
      #     #     # acc_lst0 = model0.evaluate_both(1, train_loaderA, test_loader)
      #     #     # acc_lst1 = model1.evaluate_both(1, train_loaderB, test_loader)
      #     #     # acc_lst2 = model2.evaluate_both(2,  train_loaderC, test_loader)
      #     #     # acc_lst3 = model3.evaluate_both(3, train_loaderD, test_loader)
      #     #     # acc_lst4 = model4.evaluate_both(4,  train_loaderE, test_loader)
      #     #     # acc_lst5 = model5.evaluate_both(5,  train_loaderF, test_loader)
      #     #     # acc_lst6 = model6.evaluate_both(6,  train_loaderG, test_loader)
      #     #     # acc_lst7 = model7.evaluate_both(7,  train_loaderH, test_loader)
      #     #     # acc_lst8 = model8.evaluate_both(8,  train_loaderI, test_loader)
      #     #     # acc_lst9 = model9.evaluate_both(9,  train_loaderJ, test_loader)
      #     #
      #     # # acc_lst8 = model3.evaluate_both_e2e(8, train_loader, val_loader)
      #     # print('epoch ',j );
          print('model0',acc_lst0);
          print('model1', acc_lst1);print('model2', acc_lst2);
          print('model3', acc_lst3);
          print('model4',acc_lst4);
          print('model5', acc_lst5);
          print('model6', acc_lst6);print('model7', acc_lst7);
          # print('model8', acc_lst8);print('model9', acc_lst9);
      #     # print('model8', acc_lst8)
      #     # model0.save_current_state_e2e(0, j, 200, learnrate, acc_lst0)
      #     # model1.save_current_state_e2e(1, j, 200, learnrate, acc_lst1);
      #     # model2.save_current_state_e2e(2, j, 200, learnrate, acc_lst2);
      #     # model3.save_current_state_e2e(3, j, 200, learnrate, acc_lst3);
      #     # model4.save_current_state_e2e(4, j, 200, learnrate, acc_lst4);
      #     # model5.save_current_state_e2e(5, j, 200, learnrate, acc_lst5);
      #     # model6.save_current_state_e2e(6, j, 200, learnrate, acc_lst6);
      #     # model7.save_current_state_e2e(7, j, 200, learnrate, acc_lst7);
      #     # model8.save_current_state_e2e(8, j, 200, learnrate, acc_lst8);
      #     # model9.save_current_state_e2e(9, j, 200, learnrate, acc_lst9);
          bestaccu0=save_best(0,j,acc_lst0,bestaccu0,model0);
          bestaccu1=save_best(1,j,acc_lst1,bestaccu1,model1);
          bestaccu2=save_best(2,j,acc_lst2,bestaccu2,model2);
          bestaccu3=save_best(3,j,acc_lst3,bestaccu3,model3);
          bestaccu4=save_best(4, j, acc_lst4, bestaccu4, model4);
          bestaccu5=save_best(5, j, acc_lst5, bestaccu5, model5);
          bestaccu6=save_best(6, j, acc_lst6, bestaccu6, model6);
          bestaccu7=save_best(7, j, acc_lst7, bestaccu7, model7);
      #     # bestaccu8=save_best(8, j, acc_lst8, bestaccu8, model8);
      #     # bestaccu9=save_best(9, j, acc_lst9, bestaccu9, model9)
      #     # save_best(8, j, acc_lst8, bestaccu8, model8);
