import torch
import torch.nn as nn
import my_functional as mf
import os
import torch.nn.functional as f
from helper1 import gain_schedule
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLDER = 'saved_models'
FOLDER_e2e = './saved_models_ICUB-NONiid-buffer'
# FOLDER_e2e = './test'
class MyLayer:
    def __init__(self, name, paras, stride=1, padding=0, bias=False, activations=mf.my_identity):
        self.name = name
        self.weights = 0
        self.param = paras
        self.shape = 0
        self.pre_shape = 0
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activations = activations


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class MyCNN(nn.Module):

    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.no_layers = 0
        self.layers = []
        self.no_outputs = n_classes

    def add(self, layer):
        if layer.name == 'conv':
            layer.weights = nn.Conv2d(*layer.param, stride=layer.stride, padding=layer.padding, bias=layer.bias).to(DEVICE)
            layer.activations = layer.activations
        if layer.name == 'flat':
            layer.weights = Flatten()
        if layer.name == 'pool':
            layer.weights = nn.MaxPool2d(*layer.param)
        if layer.name == 'adavg':
            layer.weights = nn.AdaptiveAvgPool2d(*layer.param)
        if layer.name == 'fc':
            layer.weights = nn.Linear(*layer.param, bias=layer.bias).to(DEVICE)
            layer.activations = layer.activations
        if layer.name == 'bn':
            layer.weights = nn.BatchNorm1d(*layer.param)
        if layer.name == 'dp':
            layer.weights = nn.Dropout(p=0.5)
        self.layers.append(layer)
        self.no_layers += 1



    def forward(self, x):
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
                # print('conv',x.shape)
            elif layer.name in ['flat', 'pool']:
                # print(layer.weights)
                 x= layer.weights(x)
                 # print('pool',x.shape)
                # print(x.shape)
            elif layer.name in ['bn' ,'dp']:
                x=layer.weights(x)
                # print('dp', x.shape)
            elif layer.name in ['identity']:
                # print(layer.weights)
                x = x
        return x

    # get shape
    def complete_net(self, train_loader):
        x = 0
        for X, y_true in train_loader:
            x = X.float()[0:1].to(DEVICE)
            print('shape',x.shape,X.shape)
            # print('xxxxxxxxxxxxxxx',x.shape, x)
#             input()
            break
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                # print(layer.name)
                # print(layer.activations, layer.weights)
                print(layer.weights(x).shape)
                x = layer.activations(layer.weights(x))
                layer.shape = x.shape[1:]

            elif layer.name in ['flat', 'pool']:
                # print(layer.name)
                # print(layer.weights)

                layer.pre_shape = x.shape[1:]
                x = layer.weights(x)
                layer.shape = x.shape[1:]
                # print(x.shape)
             # elif layer.name in ['bn']:

        return x

    # forward pass to get input of specific layer
    def forward_to_layer(self, x, to_lay):
        for layer in self.layers[0:to_lay]:
            if layer.name in ['conv', 'fc']:
                # print('x_shape', x.shape)
                # print('weight_shape', layer.weights(x).type)
                x = layer.activations(layer.weights(x))
                # print('x_activation', x.shape)
            elif layer.name in ['flat', 'pool','bn','dp']:
                # print(layer.weights)
                x = layer.weights(x)
        return x

    # backpropagate manually to know desired layer activation output
    def backward_to_layer(self, y, to_lay):
        for layer in self.layers[:-to_lay-1:-1]:
            # print('backward',layer.name,layer.weights)
            if layer.name == 'conv':
                print('conv_backward here!')
                pass
            elif layer.name == 'fc':
                y = linear_backward(y, layer.weights.weight, layer.activations, False)
            elif layer.name == 'pool':
                y = pool_backward_error(y, kernel=2)
            elif layer.name == 'flat':
                y = torch.reshape(y, torch.Size([y.shape[0]]+list(layer.pre_shape)))
        return y

    def get_weights_2(self):
        w = []
        for param in self.parameters():
            print("he he he", param)
            w.append(param.data)
        return w

    def set_weights_2(self, w):
        i = 0
        for param in self.parameters():
            # print(param)
            param.data = w[i]
            i += 1

    def get_weights(self):
        w = []
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                w.append(layer.weights.weight.data)
            elif layer.name in ['pool', 'flat']:
                w.append(0)
        return w

    def set_weights(self, w):
        i = 0
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                layer.weights.weight.data = w[i].to(DEVICE)
                i += 1
            elif layer.name in ['pool', 'flat']:
                i += 1

    def save_weights(self, path):
        w_list = self.get_weights()
        w_dict = {str(k): v for k, v in enumerate(w_list)}
        torch.save(w_dict, path)

    def load_weights(self, path):
        w_dict = torch.load(path)
        w_list = list(w_dict.values())
        self.set_weights(w_list)

    def get_weights_index(self, index):
        return self.layers[index].weights.weight.data

    def set_weights_index(self, w, index):
        self.layers[index].weights.weight.data = w


    def set_bias_index(self, w, index):
        self.layers[index].weights.bias = w

    def save_current_state_e2e(self, model_name,epoch, epochs,learnrate,acc_lst,two_lay=1):
        if two_lay == 0:
            weight_name = 'layer_wise'
        else:
            weight_name = 'two_layer'
        # if epoch - 2 != j_max:
        #     old_path = FOLDER_e2e + '/' + model_name + '_' + weight_name + '_' + str(epoch - 2)
        #     if os.path.exists(old_path):
        #         os.remove(old_path)
        # if j_max_old != j_max and j_max_old != epoch - 1:
        #     old_path = FOLDER_e2e + '/' + model_name + '_' + weight_name + '_' + str(j_max_old)
        #     if os.path.exists(old_path):
        #         os.remove(old_path)
        # path = FOLDER_e2e + '/' + model_name+'_'+ weight_name + '_' + str(epoch)
        # self.save_weights(path)
        filename = FOLDER_e2e + '/'+ 'log'  + '.txt'
        # lr=gain_schedule(epochs,epoch)*learnrate
        lr =  learnrate
        print(lr)
        if not os.path.exists(filename):
            os.mknod(filename)
        with open(filename, 'a') as out:
            if two_lay:
                out.write('R' + '\t')
            else:
                out.write('L' + '\t')
            out.write('model'+str(model_name) + '\t')
            # out.write('batch' + str(batch) + '\t')
            out.write('epoch'+str(epoch) + '\t')
            out.write('lr' + str(lr) + '\t')
            # out.write('lr' + str(epoch) + '\t')
            # out.write(str(lr) + '\t')
            out.write(str(acc_lst) + '\n')
    def save_best_accu(self, model_name,epoch, acc_lst,two_lay=1):
        filename = FOLDER_e2e + '/'+ 'log'  + '.txt'
        with open(filename, 'a') as out:
            out.write('best model'+str(model_name) + '\t')
            out.write('epoch'+str(epoch) + '\t')
            # out.write(str(lr) + '\t')
            out.write(str(acc_lst) + '\n')
    def save_current_state(self, model_name, epoch, lr, acc_lst, j_max, j_max_old, two_lay=1):
        if two_lay == 0:
            weight_name = 'layer_wise'
        else:
            weight_name = 'two_layer'
        if epoch - 2 != j_max:
            old_path = FOLDER + '/' + model_name + '_' + weight_name + '_' + str(epoch - 2)
            if os.path.exists(old_path):
                os.remove(old_path)
        if j_max_old != j_max and j_max_old != epoch - 1:
            old_path = FOLDER + '/' + model_name + '_' + weight_name + '_' + str(j_max_old)
            if os.path.exists(old_path):
                os.remove(old_path)
        path = FOLDER + '/' + model_name+'_'+ weight_name + '_' + str(epoch)
        self.save_weights(path)
        filename = FOLDER + '/' + model_name + '.txt'
        with open(filename, 'a') as out:
            if two_lay:
                out.write('R' + '\t')
            else:
                out.write('L' + '\t')
            out.write(str(epoch) + '\t')
            out.write(str(lr) + '\t')
            out.write(str(acc_lst) + '\n')

    # evaluate a fit model
    def evaluate_train(self, train_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        print('Accuracy of the network on the 50000 training images: %d %%' % (
                100 * correct_train / total_train))
        return 100 * correct_train / total_train

    # evaluate a fit model
    def evaluate_both_e2e(self, modelnum,percentage,train_loader, test_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
#                 _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        # print('Accuracy of the network of',modelnum,'on the', total_train, 'training images',
        #       100 * float(correct_train) / total_train)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            # for images, labels in test_loader:
            for i, (images, labels) in enumerate(test_loader):
                # if i ==1:
                #     print(labels)
                if i >= percentage*len(test_loader):
                    # print('now is test batch', i)
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = self(images.float())
                    _, predicted = torch.max(outputs, 1)
    #                 _, true_labels = torch.max(labels, 1)
                    true_labels = labels
                    total_test += labels.size(0)
                    correct_test += (predicted == true_labels).sum()

        # print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
        return [100 * float(correct_train) / total_train, 100 * float(correct_test) / total_test]

    def evaluate_and_filter(self, dataloader, threshold=0.9):
        self.eval()
        filtered_indices = []
        predicted_labels = []
        for index, data in enumerate(dataloader):
        # for batch in dataloader:
        #     images, labels = batch
        #     print(index)
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = self(images.float())
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # print(labels)
            for i, prob in enumerate(outputs):

                if torch.max(prob) > threshold:
                    filtered_indices.append(index)
                    predicted_labels.append(torch.argmax(prob).item())
                    # print(index,labels,outputs,torch.argmax(prob).item())
        return filtered_indices,predicted_labels

    def evaluate_both(self, modelnum, train_loader, test_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # print(images.shape)
                outputs = self(images.float())
                # print('output',outputs.shape)
                _, predicted = torch.max(outputs, 1)
                #                 _, true_labels = torch.max(labels, 1)
                # print('_', _)
                true_labels = labels
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        # print('Accuracy of the network of',modelnum,'on the', total_train, 'training images',
        #       100 * float(correct_train) / total_train)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            # for images, labels in test_loader:
            for i, (images, labels) in enumerate(test_loader):
                # if i ==1:
                #     print(labels)
                # if i >= percentage * len(test_loader):
                    # print('now is test batch', i)
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = self(images.float())
                    _, predicted = torch.max(outputs, 1)
                    #                 _, true_labels = torch.max(labels, 1)
                    true_labels = labels
                    total_test += labels.size(0)
                    correct_test += (predicted == true_labels).sum()

        # print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
        return [100 * float(correct_train) / total_train, 100 * float(correct_test) / total_test]

    def evaluate_val(self, modelnum, test_loader):


        correct_test = 0
        total_test = 0
        with torch.no_grad():
            # for images, labels in test_loader:
            for i, (images, labels) in enumerate(test_loader):
                # if i ==1:
                #     print(labels)
                # if i >= percentage * len(test_loader):
                # print('now is test batch', i)
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                #                 _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_test += labels.size(0)
                correct_test += (predicted == true_labels).sum()

        # print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
        return [ 100 * float(correct_test) / total_test]
  #   original
    # def evaluate_both(self, train_loader, test_loader):
    #     correct_train = 0
    #     total_train = 0
    #     with torch.no_grad():
    #         for  images, labels in train_loader:
    #             images = images.to(DEVICE)
    #             labels = labels.to(DEVICE)
    #             outputs = self(images.float())
    #             _, predicted = torch.max(outputs, 1)
    #             #                 _, true_labels = torch.max(labels, 1)
    #             true_labels = labels
    #             total_train += labels.size(0)
    #             correct_train += (predicted == true_labels).sum()
    #
    #     print('Accuracy of the network of on the', total_train, 'training images',
    #           100 * float(correct_train) / total_train)
    #
    #     correct_test = 0
    #     total_test = 0
    #     with torch.no_grad():
    #         for i,images, labels in test_loader:
    #             images = images.to(DEVICE)
    #             labels = labels.to(DEVICE)
    #             outputs = self(images.float())
    #             _, predicted = torch.max(outputs, 1)
    #             #                 _, true_labels = torch.max(labels, 1)
    #             true_labels = labels
    #             total_test += labels.size(0)
    #             correct_test += (predicted == true_labels).sum()
    #
    #     print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
    #     return [100 * float(correct_train) / total_train, 100 * float(correct_test) / total_test]

# def linear_backward(target, weight, func):
#     inv_f = mf.inv_fun(func)
#     return inv_f(target) @ torch.t(torch.pinverse(weight))
def linear_backward(target, weight, func, nullspace=False):
    inv_f = mf.inv_fun(func)
    if not nullspace:
        # print('no nullspace --', end=' ')
        return inv_f(target) @ torch.t(torch.pinverse(weight))
    else:
        print('v random --', end=' ')
        inv_tar = inv_f(target)
        n, _ = inv_tar.size()
        _, m = weight.size()
        I = torch.eye(m).to(DEVICE)
#         print('linear backward =====================---------------')
#         print(n, m)
        v = (1 - 2*torch.rand(n,m)).to(DEVICE)
        inv_weight = torch.pinverse(weight)
        return inv_tar @ torch.t(inv_weight) + v @ torch.t(I - inv_weight @ weight)


def pool_backward_error(target, kernel=2, method='Ave'):
    if method == 'Ave':
        return torch.repeat_interleave(torch.repeat_interleave(target, kernel, dim=2), kernel, dim=3)
    return 0



inchannel=3

def m0(N_CLASSES,train_loader,leakyrelu=1,sigmoid=0,wide=0):
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model0.add(MyLayer('pool', [2, 2]))
        model0.add(MyLayer('flat', 0, 0, 0))
        if sigmoid ==1:
            # model0.add(MyLayer('fc', [16384, 10], False, activations=torch.sigmoid))
            model0.add(MyLayer('fc', [802816, 10], False, activations=torch.sigmoid))

        else:#16384 409600
            # model0.add(MyLayer('dp', 0, 0, 0))
            model0.add(MyLayer('fc', [16384, 10], False, activations=torch.sigmoid))
    elif wide==1:
        if leakyrelu == 0:
            model0.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model0.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
        model0.add(MyLayer('pool', [(8,4 ),(8,4 )]))
        model0.add(MyLayer('flat', 0, 0, 0))
        if sigmoid == 1:
            model0.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
        else:
            model0.add(MyLayer('fc', [4096, 10], False))
    elif wide == 2:
        if leakyrelu == 0:
            model0.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model0.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
        model0.add(MyLayer('pool', [2, 2]))
        model0.add(MyLayer('flat', 0, 0, 0))
        if sigmoid == 1:
            model0.add(MyLayer('fc', [32768, 10], False, activations=torch.sigmoid))
        else:
            model0.add(MyLayer('fc', [32768, 10], False))
    model0.complete_net(train_loader)
    return model0
def li_m(N_CLASSES,train_loader):#[0,3]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [16384, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def li_m1(N_CLASSES,train_loader):#[0,2,5]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [8192, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def li_m2(N_CLASSES,train_loader):#[0,2,4,7]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [4096, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def li_m3(N_CLASSES,train_loader):#[0,2,4,5,8]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [1024*4, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def li_m4(N_CLASSES,train_loader):#[0,2,4,5,7,10]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def li_m5(N_CLASSES,train_loader):#[0,2,4,5,7,8,11]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def li_m6(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,13]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def li_m7(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, mf.my_identity))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def li_m8(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14,15]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    # model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    # model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    # model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def li_m9(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14,15,16]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m(N_CLASSES,train_loader):#[0,3]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [16384, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m1(N_CLASSES,train_loader):#[0,2,5]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [8192, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m2(N_CLASSES,train_loader):#[0,2,4,7]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [4096, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def lr_m3(N_CLASSES,train_loader):#[0,2,4,5,8]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [1024*4, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def lr_m4(N_CLASSES,train_loader):#[0,2,4,5,7,10]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0

def lr_m4_fc(N_CLASSES,train_loader):#[0,2,4,5,7,10]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m5(N_CLASSES,train_loader):#[0,2,4,5,7,8,11]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m5_fc(N_CLASSES,train_loader):#[0,2,4,5,7,8,11]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [2048, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m6(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,13]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m7(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14]
    model0 = MyCNN(N_CLASSES).to(DEVICE)


    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m8(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14,15]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
    model0.add(MyLayer('pool', [2, 2]))
    # model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    # model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    # model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def lr_m9(N_CLASSES,train_loader):#[0,2,4,5,7,8,10,11,14,15,16]
    model0 = MyCNN(N_CLASSES).to(DEVICE)
    model0.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [64, 128, 3], 1, 1, False, activations=f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.leaky_relu))
    model0.add(MyLayer('pool', [2, 2]))
    model0.add(MyLayer('flat', 0, 0, 0))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 512], False, activations=f.leaky_relu))
    model0.add(MyLayer('fc', [512, 10], False, activations=f.leaky_relu))
    model0.complete_net(train_loader)
    return model0
def m1(N_CLASSES,train_loader,leakyrelu=1,wide=0):
    model1 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model1.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model1.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model1.add(MyLayer('pool', [2, 2]))
        model1.add(MyLayer('conv', [64, 128, 3], 1, 1, False, f.leaky_relu))
        model1.add(MyLayer('pool', [2, 2]))
        model1.add(MyLayer('flat', 0, 0, 0))
        #8192 204800
        # model1.add(MyLayer('dp', 0, 0, 0))
        model1.add(MyLayer('fc', [8192, 10], False, activations=torch.sigmoid))
        # model1.add(MyLayer('fc', [401408, 10], False, activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model1.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model1.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
        # model1.add(MyLayer('pool', [8, 4]))
        model1.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model1.add(MyLayer('pool', [(8, 4), (8, 4)]))
        model1.add(MyLayer('flat', 0, 0, 0))
        model1.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model1.complete_net(train_loader)
    return model1
def m2(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model2 = MyCNN(N_CLASSES).to(DEVICE)
    if wide == 0:
        if leakyrelu == 0:
            model2.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model2.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model2.add(MyLayer('pool', [2, 2]))
        model2.add(MyLayer('conv', [64, 128, 3], 1, 1, False, f.leaky_relu))
        model2.add(MyLayer('pool', [2, 2]))
        model2.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
        model2.add(MyLayer('flat', 0, 0, 0))
        #16384 409600
        # model2.add(MyLayer('dp', 0, 0, 0))
        model2.add(MyLayer('fc', [16384, 10], False, activations=torch.sigmoid))
        # model2.add(MyLayer('fc', [802816, 10], False, activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model2.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model2.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
        # model2.add(MyLayer('pool', [2, 2]))
        model2.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model2.add(MyLayer('pool', [2, 2]))
        model2.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model2.add(MyLayer('pool', [(8, 4), (8, 4)]))
        model2.add(MyLayer('flat', 0, 0, 0))
        model2.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model2.complete_net(train_loader)
    # print(model2)
    return model2
def m3(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model3 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model3.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model3.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model3.add(MyLayer('pool', [2, 2]))
        model3.add(MyLayer('conv', [64, 128, 3], 1, 1, False, f.leaky_relu))
        model3.add(MyLayer('pool', [2, 2]))
        model3.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
        model3.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
        model3.add(MyLayer('pool', [2, 2]))
        model3.add(MyLayer('flat', 0, 0, 0))
        #4096 102400
        # model3.add(MyLayer('dp', 0, 0, 0))
        model3.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
        # model3.add(MyLayer('fc', [200704, 10], False, activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model3.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model3.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model3.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model3.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model3.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model3.add(MyLayer('pool', [(8, 8), (8, 8)]))
        model3.add(MyLayer('flat', 0, 0, 0))
        model3.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model3.complete_net(train_loader)
    # print(model3)
    return model3
    #
    # model4
def m4(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model4 = MyCNN(N_CLASSES).to(DEVICE)
    if wide==0:
        if leakyrelu == 0:
            model4.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model4.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model4.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [64, 128, 3], 1, 1, False, f.leaky_relu))
        model4.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.leaky_relu))
        model4.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.leaky_relu))
        model4.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.leaky_relu))
        model4.add(MyLayer('flat', 0, 0, 0))
        #8192 204800
        # model4.add(MyLayer('dp', 0, 0, 0))
        model4.add(MyLayer('fc', [8192, 10], False, activations=torch.sigmoid))
        # model4.add(MyLayer('fc', [401408, 10], False, activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model4.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model4.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model4.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model4.add(MyLayer('pool', [2, 2]))
        model4.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model4.add(MyLayer('pool', [(4, 4), (4, 4)]))
        model4.add(MyLayer('flat', 0, 0, 0))
        model4.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model4.complete_net(train_loader)
    # print(model4)
    return model4



# #model5
def m5(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model5 = MyCNN(N_CLASSES).to(DEVICE)
    if wide==0:
        if leakyrelu == 0:
            model5.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model5.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model5.add(MyLayer('pool', [2,2]))
        model5.add(MyLayer('conv', [64,128,3],1, 1,False, f.leaky_relu))
        model5.add(MyLayer('pool', [2,2]))
        model5.add(MyLayer('conv', [128,256,3],1, 1,False, f.leaky_relu))
        model5.add(MyLayer('conv', [256,256,3],1, 1,False, f.leaky_relu))
        model5.add(MyLayer('pool', [2,2]))
        model5.add(MyLayer('conv', [256,512,3],1, 1,False, f.leaky_relu))
        model5.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model5.add(MyLayer('pool', [2,2]))
        model5.add(MyLayer('flat', 0, 0, 0))
        #2048 51200
        # model5.add(MyLayer('dp', 0, 0, 0))
        model5.add(MyLayer('fc', [2048, 10], False,activations=torch.sigmoid))
        # model5.add(MyLayer('fc', [100352, 10], False, activations=torch.sigmoid))
    else:

        if leakyrelu == 0:
            model5.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model5.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model5.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model5.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model5.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model5.add(MyLayer('pool', [2, 2]))
        model5.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model5.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
        model5.add(MyLayer('pool', [(8, 4), (8, 4)]))
        model5.add(MyLayer('flat', 0, 0, 0))
        #4096 51200
        # model5.add(MyLayer('dp', 0, 0, 0))
        model5.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model5.complete_net(train_loader)
    # print(model5)
    return model5
# #model6
def m6(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model6 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model6.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model6.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model6.add(MyLayer('pool', [2,2]))
        model6.add(MyLayer('conv', [64,128,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('pool', [2,2]))
        model6.add(MyLayer('conv', [128,256,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('conv', [256,256,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('pool', [2,2]))
        model6.add(MyLayer('conv', [256,512,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('pool', [2,2]))
        model6.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model6.add(MyLayer('flat', 0, 0, 0))
        # model6.add(MyLayer('fc', [100352, 10], False,activations=torch.sigmoid))
        #2048 51200
        # model6.add(MyLayer('dp', 0, 0, 0))
        model6.add(MyLayer('fc', [2048, 10], False,activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model6.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model6.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model6.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model6.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model6.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model6.add(MyLayer('pool', [2, 2]))
        model6.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model6.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
        model6.add(MyLayer('pool', [2, 2]))
        model6.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))

        model6.add(MyLayer('pool', [(4, 2), (4, 2)]))
        model6.add(MyLayer('flat', 0, 0, 0))
        model6.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model6.complete_net(train_loader)
    # print(model6)
    return model6
# #model7
def m7(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model7 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model7.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model7.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model7.add(MyLayer('pool', [2,2]))
        model7.add(MyLayer('conv', [64,128,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('pool', [2,2]))
        model7.add(MyLayer('conv', [128,256,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('conv', [256,256,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('pool', [2,2]))
        model7.add(MyLayer('conv', [256,512,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('pool', [2,2]))
        model7.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model7.add(MyLayer('pool', [2,2]))
        model7.add(MyLayer('flat', 0, 0, 0))
        #512 12800
        # model7.add(MyLayer('dp', 0, 0, 0))
        model7.add(MyLayer('fc', [512 , 10], False,activations=torch.sigmoid))

        # model7.add(MyLayer('fc', [512, 10], False,activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model7.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model7.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model7.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model7.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('pool', [2, 2]))
        model7.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('pool', [2, 2]))
        model7.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model7.add(MyLayer('pool', [(4, 2), (4, 2)]))
        model7.add(MyLayer('flat', 0, 0, 0))
        model7.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model7.complete_net(train_loader)
    # print(model7)
    return model7
# #model8
def m8(N_CLASSES, train_loader,leakyrelu=1,wide=0):
    model8 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model8.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model8.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [64,128,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [128,256,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('conv', [256,256,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [256,512,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('flat', 0, 0, 0))
        #512 12800
        # model8.add(MyLayer('dp', 0, 0, 0))
        model8.add(MyLayer('fc', [512, 512], False,activations=f.leaky_relu))
        model8.add(MyLayer('dp', 0, 0, 0))
        model8.add(MyLayer('fc', [512, 10], False,activations=torch.sigmoid))
        # model8.add(MyLayer('fc', [25088, 10], False,activations=torch.sigmoid))

    else:
        if leakyrelu == 0:
            model8.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model8.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model8.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model8.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('pool', [2, 2]))
        model8.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('pool', [2, 2]))
        model8.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('pool', [2, 2]))
        model8.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model8.add(MyLayer('pool', [(2, 1), (2, 1)]))
        model8.add(MyLayer('flat', 0, 0, 0))
        model8.add(MyLayer('fc', [4096, 10], False, activations=torch.sigmoid))
    model8.complete_net(train_loader)
    # print(model8)
    return model8
def m8bn(N_CLASSES, train_loader,leakyrelu=0,wide=0):
    model8 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model8.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model8.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [64,128,3],1, 1,False, f.relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [128,256,3],1, 1,False, f.relu))
        model8.add(MyLayer('conv', [256,256,3],1, 1,False, f.relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [256,512,3],1, 1,False, f.relu))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.relu))
        model8.add(MyLayer('conv', [512,512,3],1, 1,False, f.relu))
        model8.add(MyLayer('pool', [2,2]))
        model8.add(MyLayer('flat', 0, 0, 0))
        model8.add(MyLayer('fc', [512, 512], False,activations=f.relu))
        model8.add(MyLayer('bn', [512]))
        model8.add(MyLayer('fc', [512, N_CLASSES], False,activations=f.relu))
        # model8.add(MyLayer('fc', [25088, 10], False,activations=torch.sigmoid))


    model8.complete_net(train_loader)
    # print(model8)
    return model8
# #model9
def m9(N_CLASSES, train_loader,leakyrelu=0,wide=0):
    model9 = MyCNN(N_CLASSES).to(DEVICE)
    if wide ==0:
        if leakyrelu == 0:
            model9.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model9.add(MyLayer('conv', [inchannel, 64, 3], 1, 1, False, f.leaky_relu))
        model9.add(MyLayer('pool', [2,2]))
        model9.add(MyLayer('conv', [64,128,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('pool', [2,2]))
        model9.add(MyLayer('conv', [128,256,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('conv', [256,256,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('pool', [2,2]))
        model9.add(MyLayer('conv', [256,512,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('pool', [2,2]))
        model9.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('conv', [512,512,3],1, 1,False, f.leaky_relu))
        model9.add(MyLayer('pool', [2,2]))
        model9.add(MyLayer('flat', 0, 0, 0))
        #512 12800
        # model9.add(MyLayer('dp', 0, 0, 0))
        model9.add(MyLayer('fc', [512, 512], False,activations=f.leaky_relu))
        model9.add(MyLayer('dp', 0, 0, 0))
        model9.add(MyLayer('fc', [512, 512], False,activations=f.leaky_relu))
        model9.add(MyLayer('dp', 0, 0, 0))
        # model9.add(MyLayer('fc', [512, 10], False,activations=torch.sigmoid))
        model9.add(MyLayer('fc', [512, N_CLASSES], False, activations=torch.sigmoid))
    else:
        if leakyrelu == 0:
            model9.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.relu))
        elif leakyrelu == 1:
            model9.add(MyLayer('conv', [inchannel, 128, 3], 1, 1, False, f.leaky_relu))
            # model3.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        # model3.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('conv', [128, 128, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('conv', [128, 256, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('conv', [256, 256, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('conv', [256, 512, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('conv', [512, 512, 3], 1, 1, False, f.relu))
        model9.add(MyLayer('pool', [(2, 2), (2, 2)]))
        # model10.add(MyLayer('pool', [2, 2]))
        model9.add(MyLayer('flat', 0, 0, 0))
        model9.add(MyLayer('fc', [2048, 1024], False, activations=torch.relu))
        model9.add(MyLayer('fc', [1024, 10], False, activations=torch.sigmoid))
    model9.complete_net(train_loader)
    # print(model9)
    return model9


