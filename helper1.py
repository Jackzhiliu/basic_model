import numpy as np
import time
import torch
import torch.nn.functional as f
import math
import my_functionaladj as mf
import os
import torch.nn as nn
np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE_ = ['cuda' if torch.cuda.is_available() else 'cpu']
print("→ running on (helper)", DEVICE_)
# FOLDER = 'saved_models'
# FOLDER_e2e = 'saved_models_e2e'

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def data_randomize(data, classes):
    idx = np.random.permutation(data.size()[0])
    x, y = data[idx], classes[idx]
    return x, y


def gain_schedule_old(loop, j):
    gain = 1
    if loop > 1:
        if j >= math.ceil(loop / 2):
            gain = 1 / 2
        if j >= math.ceil(3 * loop / 4) and loop > 4:
            gain = 1 / 4
        if j >= loop - 2 and loop > 5:
            gain = 1 / 20
        if j == loop - 1 and loop > 8:
            gain = 1 / 200
    return gain


def gain_schedule(loop, j):
    # gain = 1
    # if j >= math.ceil(loop / 2) and loop > 1:
    #     gain = 1 / 2
    # if j >= math.ceil(3 * loop / 4) and loop > 3:
    #     gain = 1 / 4
    # if j >= loop - 2 and loop > 11:
    #     gain = 1 / 10
    # if j == loop - 1 and loop > 12:
    #     gain = 1 / 50
    # return gain
    gain = 1
    # if j >= 30 and loop > 1:
    #     gain = 1 / 2
    # if j >= 60 and loop > 1:
    #     gain = 1 / 5
    # if j >= 180 and loop > 1:
    #     gain = 1 / 10
    # if j >= math.ceil(3 * loop / 4) and loop > 3:
    #     gain = 1 / 4
    # if j >= loop - 2 and loop > 11:
    #     gain = 1 / 10
    # if j == loop - 1 and loop > 12:
    #     gain = 1 / 50
    return gain


def my_data_loader(dataset=None, batch_size=300, shuffle=False):
    if dataset is None:
        dataset = [None, None]
    # print('shapes are:', np.shape(x1), np.shape(x2))
    shape_in = np.shape(dataset[0])
    shape_out = np.shape(dataset[1])
    if shuffle:
        print('shuffle')
        rand = np.random.permutation(shape_in[0])
    else:
        print('no_shuffle')
        rand = range(shape_in[0])
    no_batch = math.ceil(shape_in[0] / batch_size)
    data_out = []
    for i in range(no_batch):
        if (i + 1) * batch_size <= shape_in[0]:
            in_images = np.zeros((batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((batch_size, shape_out[1]))
        else:
            print(i, i * batch_size)
            in_images = np.zeros((shape_in[0] - i * batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((shape_in[0] - i * batch_size, shape_out[1]))
        for j in range(batch_size):
            # print(i, j, batch_size*i + j )
            if batch_size * i + j < shape_in[0]:
                in_images[j] = dataset[0][rand[batch_size * i + j]]
                out_labels[j] = dataset[1][rand[batch_size * i + j]]
        in_images = torch.from_numpy(in_images)
        in_images = in_images.permute(0, 3, 1, 2)
        out_labels = torch.from_numpy(out_labels)
        data_out.append([in_images, out_labels])
    return data_out


def create_matrix_x(x, _filter, stride, pad):
    # shape_x = x.shape
    # print(shape_x)
    # input()
    shape_filter = _filter.shape
    '''
    no_weights = shape_filter[2] * shape_filter[3]
    no_channels = shape_filter[1]
    out_dim = shape_x[1] - shape_filter[2] + 1
    matrix_x = torch.zeros([no_channels, no_weights, out_dim*out_dim]).to(DEVICE_[0])

    # print(out_dim)
    # input()
    for i in range(out_dim):
        for j in range(out_dim):
            temp = x[:, i:i + shape_filter[2], j:j + shape_filter[2]].to(DEVICE_[0])
            temp = temp.reshape([no_channels, no_weights, 1])
            # print(temp.shape)
            matrix_x[:,:,i*out_dim + j:i*out_dim + j+1] = temp
    # print(matrix_x.shape)

    matrix_x = matrix_x.reshape([no_channels * no_weights, out_dim * out_dim])
    # print(matrix_x.shape)
    # input()
    '''
    # print('create',x.shape)
    matrix_x = f.unfold(x, (shape_filter[2], shape_filter[3]), stride=stride, padding=pad)
    # print('create',shape_filter[2],shape_filter[3],stride,pad,matrix_x.shape)
    return matrix_x

def create_matrix_x2(x, _filter, stride, pad):
    # shape_x = x.shape
    # print(shape_x)
    # input()
    shape_filter = _filter.shape
    '''
    no_weights = shape_filter[2] * shape_filter[3]
    no_channels = shape_filter[1]
    out_dim = shape_x[1] - shape_filter[2] + 1
    matrix_x = torch.zeros([no_channels, no_weights, out_dim*out_dim]).to(DEVICE_[0])

    # print(out_dim)
    # input()
    for i in range(out_dim):
        for j in range(out_dim):
            temp = x[:, i:i + shape_filter[2], j:j + shape_filter[2]].to(DEVICE_[0])
            temp = temp.reshape([no_channels, no_weights, 1])
            # print(temp.shape)
            matrix_x[:,:,i*out_dim + j:i*out_dim + j+1] = temp
    # print(matrix_x.shape)

    matrix_x = matrix_x.reshape([no_channels * no_weights, out_dim * out_dim])
    # print(matrix_x.shape)
    # input()
    '''
    matrix_x = f.unfold(x, (3, 3), stride=stride, padding=pad)
    return matrix_x
def pool_backward_error(out_err, kernel=2, method='Ave'):
    # shape_pool = out_err.shape
    # print(shape_pool)
    # in_error = torch.zeros([shape_pool[0], shape_pool[1], shape_pool[2] * stride, shape_pool[3] * stride]).to(DEVICE_[0])
    # print('in_err',in_error.shape)
    in_error = 0
    if method == 'Ave':
        in_error = torch.repeat_interleave(torch.repeat_interleave(out_err, kernel, dim=2), kernel, dim=3)
    #     print('in_err',in_error.shape)
    #     input()
    #     for i in range(shape_pool[2] * stride):
    #         m = math.floor(i / stride)
    #         temp = out_err[:, :, m, :]
    #         for j in range(shape_pool[3] * stride):
    #             n = math.floor(j / stride)
    #             temp = out_err[:, :, m, n]
    #             if method == 'Max':
    #                 pass
    #                 # in_error(:, i, j,:) = max(temp, [], [2 3])
    #             elif method == 'Ave':
    #                 in_error[:,:, i, j] = temp
    return in_error


def sum_condition_cnn(lm=0., in_matrix=None, fc_w=None, dot_value=None, fil_w=None, pool_layer='max',
                      pool_ind=None):
    sum2 = 0
    nf, _ = fil_w.shape
    # print(nf)
    phi_s, _ = dot_value.shape
    # print(phi_s)
    size_phij = phi_s // nf
    fc_out, fc_size = fc_w.shape
    size_fc_wj = fc_size // nf
    # print('1-', nf, size_phij, size_fc_wj, fc_out)
    for j in range(nf):
        # print(j)
        phij = dot_value[j * size_phij:(j + 1) * size_phij]
        # print('phij-', dot_value.shape, phij.shape)
        fc_wj = fc_w[:, j * size_fc_wj:(j + 1) * size_fc_wj]
        # print('fc_wj-', fc_w.shape, fc_wj.shape)
        Pj = 0
        if pool_layer:
            if pool_layer == 'avg':
                Pj = lm * phij * torch.t(fc_wj)
            elif pool_layer == 'max':
                # fc_w_j_ = torch.empty([fc_out,1024])
                # for k in range(fc_out):
                #     print(k)
                fc_wj_pool_out = fc_wj
                temp = fc_wj_pool_out.shape
                # print(temp)
                fc_wj_pool_out = torch.reshape(fc_wj_pool_out,
                                               [1, fc_out, int(math.sqrt(temp[1])), int(math.sqrt(temp[1]))])
                pool_ind_ = pool_ind[:, j:j + 1, :, :]
                pool_ind_ = pool_ind_.repeat(1, fc_out, 1, 1)
                #                 print(pool_ind_, pool_ind_.shape)
                # print('before unpool', fc_wj_pool_out.shape, pool_ind_.shape)
                fc_wj_conv_out = f.max_unpool2d(fc_wj_pool_out, pool_ind_, 2)
                temp = fc_wj_conv_out.shape
                # print(temp)
                fc_w_j_ = torch.reshape(fc_wj_conv_out, [fc_out, temp[2] * temp[3]])
                #                 print(fc_w_j_)
                #                 print(fc_w_j_.shape)
                # print(fc_wj_pool_out, fc_wj_conv_out, fc_w_j_)
                #                 input()
                # print('before multi', phij.shape, fc_w_j_.shape)
                Pj = lm * phij * torch.t(fc_w_j_)
        else:
            # print('before multi', phij.shape, fc_wj.shape)
            Pj = lm * phij * torch.t(fc_wj)
        # print('Pj-', Pj.shape)
        sum2 = sum2 + torch.t(Pj) @ torch.t(in_matrix) @ in_matrix @ Pj
        # print('sum2-', sum2.shape)
        # input()
    return sum2


def inc_solve_2_layer_conv_fc(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print('shallow fil',fil.shape)

    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # print('shallow fil_w', fil_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            in_matrix = create_matrix_x(in_image[i:i+1].to(DEVICE_[0]), fil, stride, pad)[0]
            # print(out_image[i].shape)
            fc_out = out_image[i]
            # print('fc_out',fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('conv_out',conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
                pool_out_shape = pool_out.shape
                # print('pool_flatten_shape',pool_out_shape)
                pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            else:
                fc_in = torch.reshape(conv_out, conv_flat_shape)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            e_ = fc_out - y_
            # print('e_',e_.shape)
            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer
            e_fc_in = torch.t(fc_w_new) @ e_
            if pool_layer:
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                # print('shallow',e_fc_in.shape,pool_out_shape)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind,ker)
                    # print('shallow', e_conv_out.shape, e_pool_out.shape)
                e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
                # print('e_conv_out', e_conv_out.shape)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()

            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)

            # print('dot',dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('dot_value',dot_value.shape)
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # print('dot',dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('e_conv_out', e_conv_out.shape)
            e_conv_flat = dot_value * e_conv_out
            # print('shallow e_conv_flat', e_conv_flat.shape,e_conv_out.shape)
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)
            # print('shallow e_conv', e_conv.shape)
            # print(auto)
            if auto:
                # print(auto)
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (1.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2-2*lm)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after))-alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1
                eig_values1, _ = torch.linalg.eig(sum1)
                eig_values2, _ = torch.linalg.eig(sum2)
                # print(eig_values1)
                while (torch.min(eig_values1.real,eig_values2.real)< -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    sum1 = torch.diagflat(
                        (1.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                    # print(dot_value.shape, dot_value[1000:1200])
                    # input()
                    sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm, in_matrix,
                                                                                                       fc_w_new,
                                                                                                       dot_value,
                                                                                                       fil_w_new,
                                                                                                       pool_layer=pool_layer,
                                                                                                       pool_ind=pool_ind)
                    # input()
                    lr_con = sum1
                    eig_values1, _ = torch.linalg.eig(sum1)
                    eig_values2, _ = torch.linalg.eig(sum2)
                    print(alpha_v*gain,alpha_w*gain,  mf.fun_max_derivative(fun_after))
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def identity(x):
    return x


# def inc_solve_2_layer_conv_fc_acce(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
#                               fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
#                               loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
#         # print('leeeee', out_image[0])
#         out_shape = out_image.shape
#         # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#         #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
#         out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
#
#         shape_filter = fil.shape
#         # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
#         no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
#         no_fil_channels = shape_filter[0]
#         # print('shallow fil',fil.shape)
#
#         fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
#         # print('shallow fil_w', fil_w.shape)
#         fc_w = fc_wei.to(DEVICE_[0])
#
#         lm = gain
#
#         pool_ind = None
#         pool_out = None
#         if mix:
#             pass
#             input()
#         alpha_v = torch.tensor(1).to(DEVICE_[0])
#         alpha_w = torch.tensor(1).to(DEVICE_[0])
#         for j in range(loop):  # Each epoch
#             if batch_no == 0:
#                 print('= loop ', lm, ' =')
#             alpha_v = torch.tensor(1).to(DEVICE_[0])
#             alpha_w = torch.tensor(1).to(DEVICE_[0])
#
#             for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
#
#                 fil_w_new = fil_w
#                 fc_w_new = fc_w
#                 in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
#
#                 fc_out = out_image[i]
#
#                 conv_act = fil_w_new @ in_matrix  # VX
#
#                 conv_out = fun_front(conv_act)
#
#                 conv_out_shape=conv_out.shape
#
#                 conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
#                 conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
#                 # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
#                 conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
#                                                     int(math.sqrt(conv_out_shape[1]))])
#
#                 if pool_layer:
#                     if pool_layer == 'avg':
#                         pool_out = f.avg_pool2d(conv_out, 2, 2)
#                     elif pool_layer == 'max':
#                         #                     print('something')
#                         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
#                         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
#                     pool_out_shape = pool_out.shape
#                     # print('pool_flatten_shape',pool_out_shape)
#                     pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
#                     fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
#                 else:
#                     fc_in = torch.reshape(conv_out, conv_flat_shape)
#                 # print(fc_w_new.shape,fc_in.shape)
#                 y_ = fun_after(fc_w_new @ fc_in)
#                 e_ = fc_out - y_
#
#                 e_fc_in = torch.t(fc_w_new) @ e_
#                 print(e_fc_in.shape)
#                 if pool_layer:
#                     e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
#                     # print('shallow',e_fc_in.shape,pool_out_shape)
#                     # Backpropagation   to   conv     layer
#                     if pool_layer == 'avg':
#                         e_conv_out = pool_backward_error(e_pool_out, 2)
#                     elif pool_layer == 'max':
#                         e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, ker)
#                         # print('shallow', e_conv_out.shape, e_pool_out.shape)
#                     e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
#                     # print('e_conv_out', e_conv_out.shape)
#                 else:
#                     e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
#
#
#                 dot_value = mf.derivative_fun(fun_front)(conv_act_flat, slope)
#                 dot_value = dot_value.reshape(conv_flat_shape)
#                 e_conv_flat = dot_value * e_conv_out
#                 e_conv = torch.reshape(e_conv_flat, conv_out_shape)
#
#                 if auto:
#
#                     sum1 = torch.diagflat(
#                         (1.0 * lm / mf.fun_max_derivative(fun_after)
#                          - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
#                         * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
#
#                     sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm, in_matrix,
#                                                                                                        fc_w_new,
#                                                                                                        dot_value,
#                                                                                                        fil_w_new,
#                                                                                                        pool_layer=pool_layer,
#                                                                                                        pool_ind=pool_ind)
#
#                     lr_con = sum1
#                     eig_values1, _ = torch.linalg.eig(sum1)
#                     eig_values2, _ = torch.linalg.eig(sum2)
#                     # print(eig_values1)
#                     while (torch.min(eig_values1.real, eig_values2.real) < -0.005).any():
#
#                         alpha_v = alpha_v / 1.1
#                         alpha_w = alpha_w / 1.1
#                         sum1 = torch.diagflat(
#                             (1.0 * lm / mf.fun_max_derivative(fun_after)
#                              - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
#                             * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
#
#                         sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm,
#                                                                                                            in_matrix,
#                                                                                                            fc_w_new,
#                                                                                                            dot_value,
#                                                                                                            fil_w_new,
#                                                                                                            pool_layer=pool_layer,
#                                                                                                            pool_ind=pool_ind)
#                         # input()
#                         lr_con = sum1
#                         eig_values1, _ = torch.linalg.eig(sum1)
#                         eig_values2, _ = torch.linalg.eig(sum2)
#                         print(alpha_v * gain, alpha_w * gain, mf.fun_max_derivative(fun_after))
#                 fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
#                 # print(e_conv.shape,in_matrix.shape)
#                 fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
#         #
#         fil_w = torch.reshape(fil_w, shape_filter)
#         fc_w = fc_w
#         lr = alpha_v * lm
#         return fil_w, fc_w, alpha_v, lr.item(), e_


def inc_solve_2_layer_conv_fc_acce(epoch_no, batch_no, in_image, out_image, ker, stri, slope, pool_layer='max',
                                   fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                                   loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    DEVICE =  DEVICE_[0]

    out_image = torch.reshape(out_image, (out_image.shape[0], out_image.shape[1], 1)).to(DEVICE)
    fil_w = torch.reshape(fil, (fil.shape[0], -1)).to(DEVICE)
    fc_w = fc_wei.to(DEVICE)

    alpha_v = torch.tensor(1.0, device=DEVICE)
    alpha_w = torch.tensor(1.0, device=DEVICE)

    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', gain, ' =')
        alpha_v.fill_(1.0)
        alpha_w.fill_(1.0)

        for i in range(out_image.shape[0]):  # Number of data samples (output)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE), fil, stride, pad)[0]
            # print(in_matrix.shape,in_image[i:i + 1].shape)
            fc_out = out_image[i]

            conv_act = torch.matmul(fil_w, in_matrix)
            conv_out = fun_front(conv_act)
            conv_out_shape=conv_out.shape
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_out = torch.reshape(conv_out, (1, conv_out.shape[0], int(math.sqrt(conv_out.shape[1])),
                                                int(math.sqrt(conv_out.shape[1]))))
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            #     pool_out_shape = pool_out.shape
            #     # print('pool_flatten_shape',pool_out_shape)
            #     pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
            #     fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            # else:
            #     fc_in = torch.reshape(conv_out, conv_flat_shape)
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
                # print('conv_out',conv_out.shape)
                    fc_in = pool_out.reshape(-1, 1)
            else:
                 fc_in = torch.reshape(conv_out, conv_flat_shape)
            y_ = fun_after(fc_w @ fc_in)
            e_ = fc_out - y_
            # print(e_)
            e_fc_in = fc_w.t() @ e_
            # print('pool_out.shape',pool_out.shape)
            if pool_layer:
                e_pool_out = e_fc_in.reshape(pool_out.shape)
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, ker)
                e_conv_out = e_conv_out.reshape(-1, 1)
            else:
                e_conv_out = e_fc_in.reshape(-1, 1)

            dot_value = mf.derivative_fun(fun_front)(conv_act.flatten(), slope).reshape(-1, 1)
            # print(conv_act.flatten())
            # print(dot_value.shape, e_conv_out.shape)
            e_conv_flat = dot_value * e_conv_out
            # print(conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)
            lm = gain
            if i==1:
                if auto:
                    # print(auto)
                    # print(fc_w_new.shape, dot_vaclue.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                    # print(lm)
                    # print(mf.fun_max_derivative(fun_after))
                    # print(alpha_w)
                    # print(torch.sum(conv_act_flat ** 2))
                    # print(-2*e_.t()@e_ * lm+alpha_w*e_.t()@e_ * lm** 2*torch.sum(fc_in ** 2))
                    # Tt= gain * e_conv @ in_matrix.t()
                    # print(Tt.shape)
                    # print(-2*e_.t()@e_ * lm+alpha_v *torch.sum(torch.sum(Tt**2, dim=0)))
                    sum=-2*e_.t()@e_ * lm+alpha_w*e_.t()@e_ * lm** 2*torch.sum(fc_in ** 2)+-2*e_.t()@e_ * lm+alpha_v *torch.sum(torch.sum((gain * e_conv @ in_matrix.t())**2, dim=0))
                    # print(sum<0)
                    while sum>0:
                        alpha_v /= 1.1
                        alpha_w /= 1.1
                        sum = -2 * e_.t() @ e_ * lm + alpha_w * e_.t() @ e_ * lm ** 2 * torch.sum(
                            fc_in ** 2) + -2 * e_.t() @ e_ * lm + alpha_v * torch.sum(torch.sum((gain * e_conv @ in_matrix.t())** 2, dim=0))
                        if sum < 0:
                            break
                # print(-2 * e_.t() @ e_ * lm + alpha_w * e_.t() @ e_ * lm ** 2 * torch.sum(fc_in ** 2))
                # sum1 = torch.diagflat(
                #     (1.0 * lm / mf.fun_max_derivative(fun_after)
                #      - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
                #     * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                #
                # # print(dot_value.shape, dot_value[1000:1200])
                # # input()
                # sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm, in_matrix,
                #                                                                                    fc_w_new, dot_value,
                #                                                                                    fil_w_new,
                #                                                                                    pool_layer=pool_layer,
                #                                                                                    pool_ind=pool_ind)
                # # input()
                # lr_con = sum1
                # eig_values1, _ = torch.linalg.eig(sum1)
                # eig_values2, _ = torch.linalg.eig(sum2)
                # # print(eig_values1)
                # while (torch.min(eig_values1.real, eig_values2.real) < -0.005).any():
                #     # print('%d - %d', j, i)
                #     #                 print(eig_values[:, 0])
                #     alpha_v = alpha_v / 1.1
                #     alpha_w = alpha_w / 1.1
                #     sum1 = torch.diagflat(
                #         (1.0 * lm / mf.fun_max_derivative(fun_after)
                #          - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
                #         * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                #
                #     # print(dot_value.shape, dot_value[1000:1200])
                #     # input()
                #     sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm, in_matrix,
                #                                                                                        fc_w_new,
                #                                                                                        dot_value,
                #                                                                                        fil_w_new,
                #                                                                                        pool_layer=pool_layer,
                #                                                                                        pool_ind=pool_ind)
                #     # input()
                #     lr_con = sum1
                #     eig_values1, _ = torch.linalg.eig(sum1)
                #     eig_values2, _ = torch.linalg.eig(sum2)
                #     print(alpha_v * gain, alpha_w * gain, mf.fun_max_derivative(fun_after))
            # if auto:
            #     sum1 = torch.diagflat(
            #         (1.0 * gain / mf.fun_max_derivative(fun_after) -
            #          alpha_w * torch.sum(conv_act.flatten() ** 2).to(DEVICE) * gain ** 2 - 2 * gain) *
            #         torch.ones(out_image.shape[1], 1, device=DEVICE))
            #
            #     sum2 = (1.0 * gain / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(
            #         gain, in_matrix, fc_w, dot_value, fil_w, pool_layer=pool_layer, pool_ind=pool_ind)
            #
            #     eig_values1 = torch.linalg.eigvals(sum1).real
            #     eig_values2 = torch.linalg.eigvals(sum2).real
            #
            #     while torch.min(eig_values1, eig_values2) < -0.005:
            #         alpha_v /= 1.1
            #         alpha_w /= 1.1
            #         sum1 = torch.diagflat(
            #             (1.0 * gain / mf.fun_max_derivative(fun_after) -
            #              alpha_w * torch.sum(conv_act.flatten() ** 2).to(DEVICE) * gain ** 2 - 2 * gain) *
            #             torch.ones(out_image.shape[1], 1, device=DEVICE))
            #         sum2 = (1.0 * gain / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(
            #             gain, in_matrix, fc_w, dot_value, fil_w, pool_layer=pool_layer, pool_ind=pool_ind)
            #
            #         eig_values1 = torch.linalg.eigvals(sum1).real
            #         eig_values2 = torch.linalg.eigvals(sum2).real
            #
            #         # print(alpha_v * gain, alpha_w * gain, mf.fun_max_derivative(fun_after))
            # print(fc_in.shape)
            fc_w += alpha_w * gain * e_ @ fc_in.t()
            fil_w += alpha_v * gain * e_conv @ in_matrix.t()
            # print(fil_w.shape,fil.shape)
            # print(e_conv.shape,in_matrix.shape)
    fil_w = fil_w.reshape(fil.shape)

    lr = alpha_v * gain
    return fil_w, fc_w, alpha_v, lr.item(), e_


# def inc_solve_2_layer_conv_fc_acce(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
#                               fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
#                               loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
#         # print('leeeee', out_image[0])
#         out_shape = out_image.shape
#         # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#         #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
#         out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
#
#         shape_filter = fil.shape
#         # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
#         no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
#         no_fil_channels = shape_filter[0]
#         # print('shallow fil',fil.shape)
#
#         fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
#         # print('shallow fil_w', fil_w.shape)
#         fc_w = fc_wei.to(DEVICE_[0])
#
#         lm = gain
#
#         pool_ind = None
#         pool_out = None
#         if mix:
#             pass
#             input()
#         alpha_v = torch.tensor(1).to(DEVICE_[0])
#         alpha_w = torch.tensor(1).to(DEVICE_[0])
#         for j in range(loop):  # Each epoch
#             if batch_no == 0:
#                 print('= loop ', lm, ' =')
#             alpha_v = torch.tensor(1).to(DEVICE_[0])
#             alpha_w = torch.tensor(1).to(DEVICE_[0])
#
#             for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
#
#                 fil_w_new = fil_w
#                 fc_w_new = fc_w
#                 in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
#
#                 fc_out = out_image[i]
#
#                 conv_act = torch.matmul(fil_w, in_matrix)
#
#                 conv_out = fun_front(conv_act)
#
#                 conv_out_shape=conv_out.shape
#
#                 conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
#                 conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
#                 # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
#                 conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
#                                                     int(math.sqrt(conv_out_shape[1]))])
#
#                 if pool_layer:
#                     if pool_layer == 'avg':
#                         pool_out = f.avg_pool2d(conv_out, 2, 2)
#                     elif pool_layer == 'max':
#                         #                     print('something')
#                         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
#                         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
#                     pool_out_shape = pool_out.shape
#                     # print('pool_flatten_shape',pool_out_shape)
#                     pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
#                     fc_in = pool_out.reshape(-1, 1)
#                 else:
#                     fc_in = torch.reshape(conv_out, conv_flat_shape)
#                 # print(fc_w_new.shape,fc_in.shape)
#                 y_ = fun_after(fc_w_new @ fc_in)
#                 e_ = fc_out - y_
#
#                 e_fc_in = torch.t(fc_w_new) @ e_
#                 if pool_layer:
#                     e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
#                     # print('shallow',e_fc_in.shape,pool_out_shape)
#                     # Backpropagation   to   conv     layer
#                     if pool_layer == 'avg':
#                         e_conv_out = pool_backward_error(e_pool_out, 2)
#                     elif pool_layer == 'max':
#                         e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, ker)
#                         # print('shallow', e_conv_out.shape, e_pool_out.shape)
#                     e_conv_out = e_conv_out.reshape(-1, 1)
#                     # print('e_conv_out', e_conv_out.shape)
#                 else:
#                     e_conv_out = e_conv_out.reshape(-1, 1)
#
#                 dot_value = mf.derivative_fun(fun_front)(conv_act.flatten(), slope).reshape(-1, 1)
#
#                 e_conv_flat = dot_value * e_conv_out
#                 e_conv = torch.reshape(e_conv_flat, conv_out_shape)
#
#                 if auto:
#
#                     sum1 = torch.diagflat(
#                         (1.0 * lm / mf.fun_max_derivative(fun_after)
#                          - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
#                         * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
#
#                     sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm, in_matrix,
#                                                                                                        fc_w_new,
#                                                                                                        dot_value,
#                                                                                                        fil_w_new,
#                                                                                                        pool_layer=pool_layer,
#                                                                                                        pool_ind=pool_ind)
#
#                     lr_con = sum1
#                     eig_values1, _ = torch.linalg.eig(sum1)
#                     eig_values2, _ = torch.linalg.eig(sum2)
#                     # print(eig_values1)
#                     while (torch.min(eig_values1.real, eig_values2.real) < -0.005).any():
#
#                         alpha_v = alpha_v / 1.1
#                         alpha_w = alpha_w / 1.1
#                         sum1 = torch.diagflat(
#                             (1.0 * lm / mf.fun_max_derivative(fun_after)
#                              - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2 - 2 * lm)
#                             * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
#
#                         sum2 = (1.0 * lm / mf.fun_max_derivative(fun_after)) - alpha_v * sum_condition_cnn(lm,
#                                                                                                            in_matrix,
#                                                                                                            fc_w_new,
#                                                                                                            dot_value,
#                                                                                                            fil_w_new,
#                                                                                                            pool_layer=pool_layer,
#                                                                                                            pool_ind=pool_ind)
#                         # input()
#                         lr_con = sum1
#                         eig_values1, _ = torch.linalg.eig(sum1)
#                         eig_values2, _ = torch.linalg.eig(sum2)
#                         print(alpha_v * gain, alpha_w * gain, mf.fun_max_derivative(fun_after))
#                 fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
#                 # print(e_conv.shape,in_matrix.shape)
#                 fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
#         #
#         fil_w = torch.reshape(fil_w, shape_filter)
#         fc_w = fc_w
#         lr = alpha_v * lm
#         return fil_w, fc_w, alpha_v, lr.item(), e_
def inc_solve_2_layer_conv_fc_inverse(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None, filnext=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print('shallow fil',fil.shape)

    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # print('shallow fil_w', fil_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            in_matrix = create_matrix_x(in_image[i:i+1].to(DEVICE_[0]), fil, stride, pad)[0]
            # print(out_image[i].shape)
            fc_out = out_image[i]
            print('fc_out',fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape = conv_out.shape
            print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            print('conv_out',conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
                pool_out_shape = pool_out.shape
                # print('pool_flatten_shape',pool_out_shape)
                pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            else:
                fc_in = torch.reshape(conv_out, conv_flat_shape)
            print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            e_ = fc_out - y_
            print('e_',e_.shape)
            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer
            e_fc_in = torch.t(fc_w_new) @ e_
            if pool_layer:
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                # print('shallow',e_fc_in.shape,pool_out_shape)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind,ker)
                    # print('shallow', e_conv_out.shape, e_pool_out.shape)
                e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
                # print('e_conv_out', e_conv_out.shape)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()

            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)

            # print('dot',dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('dot_value',dot_value.shape)
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # print('dot',dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('e_conv_out', e_conv_out.shape)
            e_conv_flat = dot_value * e_conv_out
            # print('shallow e_conv_flat', e_conv_flat.shape,e_conv_out.shape)
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)
            # print('shallow e_conv', e_conv.shape)
            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix @ filnext @ torch.inverse(fil_w_new))
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_

def inc_solve_2_layer_conv_fc_batch(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[1], out_shape[0]]).to(DEVICE_[0])
    # out_image=out_image.to(DEVICE_[0])
    # print(out_image.shape)
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print('shallow fil',fil.shape)

    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # print('shallow fil_w', fil_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

    # for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
        # print(i, out_shape[0])
        # t0 = time.time()
        fil_w_new = fil_w
        fc_w_new = fc_w
        in_matrix = create_matrix_x(in_image.to(DEVICE_[0]), fil, stride, pad)
        # print(out_image[i].shape)
        fc_out = out_image
        # print('fc_out',fc_out.shape)
        conv_act = fil_w_new @ in_matrix# VX
        conv_out = fun_front(conv_act)
        # print(conv_act)
        # print(conv_out)
        #             t1 = time.time()
        #             print('create matrix time', t1-t0)

        conv_out_shape = conv_out.shape
        # print('conv_out_shape', conv_out_shape)
        # input()
        # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
        conv_flat_shape = [conv_out_shape[1] * conv_out_shape[2],  conv_out_shape[0]]
        conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
        # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
        conv_out = torch.reshape(conv_out, [conv_out_shape[0], conv_out_shape[1], int(math.sqrt(conv_out_shape[2])),
                                            int(math.sqrt(conv_out_shape[2]))])
        # print('conv_out',conv_out.shape)
        # Apply     pooling        layer
        if pool_layer:
            if pool_layer == 'avg':
                pool_out = f.avg_pool2d(conv_out, 2, 2)
            elif pool_layer == 'max':
                #                     print('something')
                # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            pool_out_shape = pool_out.shape
            # print(pool_out_shape)
            pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3]
            fc_in = torch.reshape(pool_out, [pool_flatten_shape, pool_out_shape[0]])
        else:
            fc_in = torch.reshape(conv_out, conv_flat_shape)
        # print(fc_w_new.shape,fc_in.shape)
        y_ = fun_after(fc_w_new @ fc_in)
        e_ = fc_out - y_
        # print('e_',e_.shape)
        # print('y_', y_.shape)
        #             t2 = time.time()
        #             print('update fc_w time', t2-t1)
        # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

        # Backpropagation  to  flattening & pooling   layer
        e_fc_in = torch.t(fc_w_new) @ e_
        if pool_layer:
            e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
            # print('shallow',e_fc_in.shape,pool_out_shape)
            # Backpropagation   to   conv     layer
            if pool_layer == 'avg':
                e_conv_out = pool_backward_error(e_pool_out, 2)
            elif pool_layer == 'max':
                e_conv_out = f.max_unpool2d(e_pool_out, pool_ind,ker)
                # print('shallow', e_conv_out.shape, e_pool_out.shape)
            e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
            # print('e_conv_out', e_conv_out.shape)
        else:
            e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
        # input()
        # print('deri start')
        # t0 = time.time()

        dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)

        # print('dot',dot_value.shape)
        # print(fun_front)
        # print(mf.derivative_fun(fun_front))
        # print(conv_act_flat)
        # print(dot_value)
        # t1 = time.time()
        # print('derivative time', t1-t0)
        # print('dot_value',dot_value.shape)
        dot_value = dot_value.reshape(conv_flat_shape)
        # print('dot_value', dot_value.shape)
        # print('dot',dot_value.shape)
        # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
        #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
        #             input()
        # print('e_conv_out', e_conv_out.shape)
        e_conv_flat = dot_value * e_conv_out
        # print('shallow e_conv_flat', e_conv_flat.shape,e_conv_out.shape)
        #             for k in range(conv_flat_shape[0]):
        #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
        # if deri_conv_act[k] != 0:
        #     e_conv_flat[k] = e_fc_in[k]
        # else:
        #     e_conv_flat[k] = 0
        #             t3 = time.time()
        #             print('diagonal time', t3 - t2)
        # print(e_conv_flat.shape, conv_out_shape)
        e_conv = torch.reshape(e_conv_flat, conv_out_shape)
        # print('e_conv',e_conv.shape)
        # print('shallow e_conv', e_conv.shape)
        if auto:
            # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
            # print(lm)
            # print(mf.fun_max_derivative(fun_after))
            # print(alpha_w)
            # print(torch.sum(conv_act_flat ** 2))
            sum1 = torch.diagflat(
                (2.0 * lm / mf.fun_max_derivative(fun_after)
                 - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

            # print(dot_value.shape, dot_value[1000:1200])
            # input()
            sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                               pool_ind=pool_ind)
            # input()
            lr_con = sum1 - sum2
            eig_values, _ = torch.linalg.eig(lr_con)
            while (eig_values[:, 0] < -0.005).any():
                # print('%d - %d', j, i)
                #                 print(eig_values[:, 0])
                alpha_v = alpha_v / 1.1
                alpha_w = alpha_w / 1.1
                # print(alpha_v)
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                   pool_layer=pool_layer, pool_ind=pool_ind)
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
        fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
        # transpose
        # print('inmatrix',torch.transpose(in_matrix,1,2).shape)
        fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.transpose(in_matrix,1,2)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()
    # print(fil_w.shape)
    # print(fil_w_new.shape)
    fil_w=torch.mean(fil_w, 0)
    # print(fil_w.shape)
    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_bn(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.], n 43r
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print('shallow fil',fil.shape)
    # print('fil shape',fil.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # print('shallow fil_w', fil_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
            in_matrix_batch = create_matrix_x(in_image.to(DEVICE_[0]), fil, stride, pad)
            # print(out_image[i].shape)
            fc_out = out_image[i]
            # conv_actww = fil_w_new @ in_image
            # print('image',in_image.shape)
            # print('fil_w_new', fil_w_new.shape)
            # print('in_matrix', in_matrix.shape)
            conv_act = fil_w_new @ in_matrix  # VX
            # print('conv_act',conv_act.shape)
            conv_act_batch = fil_w_new @ in_matrix_batch  # VX
            bn=nn.BatchNorm1d(64).to(DEVICE_[0])

            # conv_act_batch=conv_act_batch.unsqueeze(0)
            # print(conv_act_batch.shape)
            conv_act=bn(conv_act_batch)
            conv_act = conv_act[i]
            # print('bn',conv_act.shape)
            conv_out = fun_front(conv_act)
            # print(conv_act)
            # print(fun_front)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])

            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
                pool_out_shape = pool_out.shape
                pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            else:
                fc_in = torch.reshape(conv_out, conv_flat_shape)
            y_ = fun_after(fc_w_new @ fc_in)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer
            e_fc_in = torch.t(fc_w_new) @ e_
            if pool_layer:
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                # print('shallow',e_fc_in.shape,pool_out_shape)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, ker)
                    # print('shallow', e_conv_out.shape, e_pool_out.shape)
                e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
                # print('e_conv_out', e_conv_out.shape)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()

            dot_value = mf.derivative_fun(fun_front)(conv_act_flat, slope)

            # print('dot',dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('dot_value',dot_value.shape)
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # print('dot',dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('e_conv_out', e_conv_out.shape)
            e_conv_flat = dot_value * e_conv_out
            # print('shallow e_conv_flat', e_conv_flat.shape,e_conv_out.shape)
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)
            # print('shallow e_conv', e_conv.shape)
            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(), e_

def inc_solve_2_layer_conv_fc_bn_batch(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[1], out_shape[0]]).to(DEVICE_[0])
    # out_image=out_image.to(DEVICE_[0])
    # print(out_image.shape)
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print('shallow fil',fil.shape)

    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # print('shallow fil_w', fil_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        # for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
        # print(i, out_shape[0])
        # t0 = time.time()
        fil_w_new = fil_w
        fc_w_new = fc_w
        in_matrix = create_matrix_x(in_image.to(DEVICE_[0]), fil, stride, pad)
        # print(out_image[i].shape)
        fc_out = out_image
        # print('fc_out', fc_out.shape)
        conv_act = fil_w_new @ in_matrix  # VX
        bn = nn.BatchNorm1d(shape_filter[0]).to(DEVICE_[0])
        conv_act = bn(conv_act)
        conv_out = fun_front(conv_act)
        # print(conv_act)
        # print(conv_out)
        #             t1 = time.time()
        #             print('create matrix time', t1-t0)

        conv_out_shape = conv_out.shape
        # print('conv_out_shape', conv_out_shape)
        # input()
        # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
        conv_flat_shape = [conv_out_shape[1] * conv_out_shape[2], conv_out_shape[0]]
        conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
        # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
        conv_out = torch.reshape(conv_out, [conv_out_shape[0], conv_out_shape[1], int(math.sqrt(conv_out_shape[2])),
                                            int(math.sqrt(conv_out_shape[2]))])
        # print('conv_out', conv_out.shape)
        # Apply     pooling        layer
        if pool_layer:
            if pool_layer == 'avg':
                pool_out = f.avg_pool2d(conv_out, 2, 2)
            elif pool_layer == 'max':
                #                     print('something')
                # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            pool_out_shape = pool_out.shape
            # print(pool_out_shape)
            pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3]
            fc_in = torch.reshape(pool_out, [pool_flatten_shape, pool_out_shape[0]])
        else:
            fc_in = torch.reshape(conv_out, conv_flat_shape)
        # print(fc_w_new.shape, fc_in.shape)
        y_ = fun_after(fc_w_new @ fc_in)
        e_ = fc_out - y_
        # print('e_', e_.shape)
        # print('y_', y_.shape)
        #             t2 = time.time()
        #             print('update fc_w time', t2-t1)
        # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

        # Backpropagation  to  flattening & pooling   layer
        e_fc_in = torch.t(fc_w_new) @ e_
        if pool_layer:
            e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
            # print('shallow',e_fc_in.shape,pool_out_shape)
            # Backpropagation   to   conv     layer
            if pool_layer == 'avg':
                e_conv_out = pool_backward_error(e_pool_out, 2)
            elif pool_layer == 'max':
                e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, ker)
                # print('shallow', e_conv_out.shape, e_pool_out.shape)
            e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
            # print('e_conv_out', e_conv_out.shape)
        else:
            e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
        # input()
        # print('deri start')
        # t0 = time.time()

        dot_value = mf.derivative_fun(fun_front)(conv_act_flat, slope)

        # print('dot',dot_value.shape)
        # print(fun_front)
        # print(mf.derivative_fun(fun_front))
        # print(conv_act_flat)
        # print(dot_value)
        # t1 = time.time()
        # print('derivative time', t1-t0)
        # print('dot_value',dot_value.shape)
        dot_value = dot_value.reshape(conv_flat_shape)
        # print('dot_value', dot_value.shape)
        # print('dot',dot_value.shape)
        # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
        #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
        #             input()
        # print('e_conv_out', e_conv_out.shape)
        e_conv_flat = dot_value * e_conv_out
        # print('shallow e_conv_flat', e_conv_flat.shape,e_conv_out.shape)
        #             for k in range(conv_flat_shape[0]):
        #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
        # if deri_conv_act[k] != 0:
        #     e_conv_flat[k] = e_fc_in[k]
        # else:
        #     e_conv_flat[k] = 0
        #             t3 = time.time()
        #             print('diagonal time', t3 - t2)
        # print(e_conv_flat.shape, conv_out_shape)
        e_conv = torch.reshape(e_conv_flat, conv_out_shape)
        # print('e_conv', e_conv.shape)
        # print('shallow e_conv', e_conv.shape)
        if auto:
            # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
            # print(lm)
            # print(mf.fun_max_derivative(fun_after))
            # print(alpha_w)
            # print(torch.sum(conv_act_flat ** 2))
            sum1 = torch.diagflat(
                (2.0 * lm / mf.fun_max_derivative(fun_after)
                 - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

            # print(dot_value.shape, dot_value[1000:1200])
            # input()
            sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                               pool_ind=pool_ind)
            # input()
            lr_con = sum1 - sum2
            eig_values, _ = torch.linalg.eig(lr_con)
            while (eig_values[:, 0] < -0.005).any():
                # print('%d - %d', j, i)
                #                 print(eig_values[:, 0])
                alpha_v = alpha_v / 1.1
                alpha_w = alpha_w / 1.1
                # print(alpha_v)
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                   pool_layer=pool_layer, pool_ind=pool_ind)
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
        fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
        # transpose
        # print('inmatrix', torch.transpose(in_matrix, 1, 2).shape)
        fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.transpose(in_matrix, 1, 2)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()
    # print(fil_w.shape)
    # print(fil_w_new.shape)
    fil_w = torch.mean(fil_w, 0)
    # print(fil_w.shape)
    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm

    return fil_w, fc_w, alpha_v, lr.item(), e_

def inc_solve_2_layer_conv_fc_first(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    filafter_w= torch.reshape(filafter, [128, 64*9]).to(DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape2)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out2 = f.max_unpool2d(e_pool_out, pool_ind2,ker)
                    # print('deep', e_conv_out2.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print(pool_ind.shape)
                    e_conv_out2_reshape = torch.reshape(e_conv_out2, [128,256])
                    e_conv1=torch.t(filafter_w) @ e_conv_out2_reshape
                    e_conv1=torch.reshape(e_conv1,[1,64*9,16*16])
                    e_conv1_fold=f.fold(e_conv1,output_size=(16,16),kernel_size=3,stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv1_fold_unmax,[65536,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_


def inc_solve_2_layer_conv_fc_model2first(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None,filafter2=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]
            conv_act3= filafter2_w @ in_matrix3
            conv_out3 = fun_front(conv_act3)
            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out3, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape3)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind3,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [256,64])
                    e_conv3=torch.t(filafter2_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,1152,8*8])
                    e_conv3_fold=f.fold(e_conv3,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_unmax = f.max_unpool2d(e_conv3_fold, pool_ind2, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    e_conv1 = torch.t(filafter_w) @ e_conv_out2_reshape
                    e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv1_fold_unmax,[65536,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model2deep(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    # shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    # filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])

            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape2)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    # print(e_pool_out.shape,pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind2,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [256,64])
                    e_conv3=torch.t(filafter_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,1152,8*8])
                    e_conv3_fold=f.fold(e_conv3,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_unmax = f.max_unpool2d(e_conv3_fold, pool_ind, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    # e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    # e_conv1 = torch.t(fil_w) @ e_conv_out2_reshape
                    # e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)

            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv3_fold_unmax,[32768,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model3first(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None,filafter2=None, filafter3=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    # print()
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_w_new.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape3)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind3,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [256,64])
                    e_conv3=torch.t(filafter3_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,2304,8*8])
                    e_conv3_fold=f.fold(e_conv3,output_size=(8,8),kernel_size=3,stride=1, padding=1)

                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold,[256,64])
                    e_conv4=torch.t(filafter2_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,1152,8*8])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind2, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    e_conv1 = torch.t(filafter_w) @ e_conv_out2_reshape
                    e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv1_fold_unmax,[65536,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model3second(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              filafter=None,filafter2=None, filafter3=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # shape_filter = fil.shape
    #
    # # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    # no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    # no_fil_channels = shape_filter[0]
    # # print(no_fil_channels,no_fil_weights)
    # # print('fil',fil.shape,'filafter',filafter.shape)
    # fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    # print()
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = filafter_w
            fc_w_new = fc_w
            fc_out = out_image[i]
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix2 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter, stride, pad)[0]


            # in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_w_new.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape3)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind3,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [256,64])
                    e_conv3=torch.t(filafter3_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,2304,8*8])
                    e_conv3_fold=f.fold(e_conv3,output_size=(8,8),kernel_size=3,stride=1, padding=1)

                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold,[256,64])
                    e_conv4=torch.t(filafter2_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,1152,8*8])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind2, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    # e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    # e_conv1 = torch.t(filafter_w) @ e_conv_out2_reshape
                    # e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape2)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat2,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape2)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape)
            e_conv_out2=torch.reshape(e_conv3_fold_unmax,dot_value.shape)
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape2)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix2)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filafter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model3deep(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    # shape_filafter2 = filafter2.shape
    # # print(shape_filafter2)
    # filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # # print(filafter2_w.shape)
    # shape_filafter3 = filafter3.shape
    # # print(shape_filafter2)
    # filafter3_w = torch.reshape(filafter3,
    #                             [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
    #     DEVICE_[0])
    # print()
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # print(conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(conv_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])

            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape2)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind2,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [256,64])
                    e_conv3=torch.t(filafter_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,2304,8*8])
                    e_conv3_fold=f.fold(e_conv3,output_size=(8,8),kernel_size=3,stride=1, padding=1)

                    # e_conv3_fold_reshape=torch.reshape(e_conv3_fold,[256,64])
                    # e_conv4=torch.t(fil_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    # e_conv4=torch.reshape(e_conv4,[1,1152,8*8])
                    # e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind, ker)
                    # print('deep', e_conv4_fold.shape)
                    # e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    # e_conv1 = torch.t(filafter_w) @ e_conv_out2_reshape
                    # e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape)
            e_conv_out2=torch.reshape(e_conv3_fold,[16384,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model5deep(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    # shape_filafter2 = filafter2.shape
    # # print(shape_filafter2)
    # filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # # print(filafter2_w.shape)
    # shape_filafter3 = filafter3.shape
    # # print(shape_filafter2)
    # filafter3_w = torch.reshape(filafter3,
    #                             [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
    #     DEVICE_[0])
    # print()
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # print(conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(conv_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])

            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape2)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind2,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(pool_ind.shape)
                    e_conv_out3_shape=e_conv_out3.shape
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [e_conv_out3_shape[1],e_conv_out3_shape[2]*e_conv_out3_shape[3]])
                    e_conv3=torch.t(filafter_w) @ e_conv_out3_reshape
                    e_conv3_shape=e_conv3.shape
                    e_conv3=torch.reshape(e_conv3,[1,e_conv3_shape[0],e_conv3_shape[1]])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)

                    # e_conv3_fold_reshape=torch.reshape(e_conv3_fold,[256,64])
                    # e_conv4=torch.t(fil_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    # e_conv4=torch.reshape(e_conv4,[1,1152,8*8])
                    # e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind, ker)
                    # print('deep', e_conv4_fold.shape)
                    # e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    # e_conv1 = torch.t(filafter_w) @ e_conv_out2_reshape
                    # e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)
                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape)
            e_conv_out2=torch.reshape(e_conv3_fold,[8192,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model4deep(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    # shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    # filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])

            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape2)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    # print(e_pool_out.shape,pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind2,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,16])
                    e_conv3=torch.t(filafter_w) @ e_conv_out3_reshape
                    # print('deep', e_conv3.shape)
                    e_conv3=torch.reshape(e_conv3,[1,2304,4*4])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_unmax = f.max_unpool2d(e_conv3_fold, pool_ind, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    # e_conv_out2_reshape = torch.reshape(e_conv3_fold_unmax, [128, 256])
                    # e_conv1 = torch.t(fil_w) @ e_conv_out2_reshape
                    # e_conv1 = torch.reshape(e_conv1, [1, 64 * 9, 16 * 16])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind, ker)

            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv3_fold_unmax,[16384,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_

def inc_solve_2_layer_conv_fc_model4first(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None,filafter2=None, filafter3=None,filafter4=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out5, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter4_w) @ e_conv_out3_reshape

                    e_conv3=torch.reshape(e_conv3,[1,2304,4*4])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv3_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)

                    e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape

                    e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # print('deep', e_conv0.shape)
                    e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv0_fold_unmax,[65536,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model4second(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              filafter=None,filafter2=None, filafter3=None,filafter4=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    # no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    # no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    # fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = filafter_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix2 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            # conv_act = fil_w_new @ in_matrix# VX
            # conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            # conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            # conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
            #                                     int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            # in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out5, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter4_w) @ e_conv_out3_reshape

                    e_conv3=torch.reshape(e_conv3,[1,2304,4*4])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv3_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)

                    e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape

                    e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    # e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    # e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # # print('deep', e_conv0.shape)
                    # e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    # e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape2)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat2,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape2)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv1_fold_unmax,dot_value.shape)
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape2)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix2)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filafter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model4third(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                         filafter2=None, filafter3=None,filafter4=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    # no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    # no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    # fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # shape_filafter = filafter.shape
    # print(shape_filafter)
    # filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = filafter2_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            # in_matrix2 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter2_w, stride, pad)[0]
            #
            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            # conv_act = fil_w_new @ in_matrix# VX
            # conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            # conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            # conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
            #                                     int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            # in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            # conv_act2=filafter_w@in_matrix2
            # conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            # conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            # conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
            #                                     int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out2, 2, 2)
            #     elif pool_layer == 'max':
            #                             print('something')
            #         pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    # pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                # pool_out_shape2 = pool_out2.shape
                # pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            # else:
            #     fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out4, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out4, conv_flat_shape4)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out5, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out5, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out5, conv_flat_shape5)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter4_w) @ e_conv_out3_reshape

                    e_conv3=torch.reshape(e_conv3,[1,2304,4*4])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv3_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    # e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    # e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # # print('deep', e_conv4.shape)
                    # e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    # e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)
                    #
                    # e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    # e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape
                    #
                    # e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    # e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    # e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # # print('deep', e_conv0.shape)
                    # e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    # e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape3)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat3,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape3)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv3_fold_unmax,dot_value.shape)
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape3)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat3 ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix3, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat3 ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix3, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix3)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filafter2)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model5first(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                              fil=None,filafter=None,filafter2=None, filafter3=None,filafter4=None,filafter5=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    shape_filafter5 = filafter5.shape
    # print(shape_filafter2)
    filafter5_w = torch.reshape(filafter5,
                                [shape_filafter5[0], shape_filafter5[1] * shape_filafter5[2] * shape_filafter5[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            pool_out_shape1 = pool_out.shape
            pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            in_matrix6 = create_matrix_x(conv_out5.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act6 = filafter5_w @ in_matrix6
            conv_out6 = fun_front(conv_act6)
            conv_out_shape6 = conv_out6.shape
            conv_flat_shape6 = [conv_out_shape6[0] * conv_out_shape6[1], 1]
            conv_act_flat6 = torch.reshape(conv_act6, conv_flat_shape6)
            conv_out6 = torch.reshape(conv_out6, [1, conv_out_shape6[0], int(math.sqrt(conv_out_shape6[1])),
                                                  int(math.sqrt(conv_out_shape6[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out6, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter5_w) @ e_conv_out3_reshape

                    e_conv3_shape=e_conv3.shape
                    e_conv3=torch.reshape(e_conv3,[1,e_conv3_shape[0],e_conv3_shape[1]])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_shape = e_conv3_fold.shape
                    # print(e_conv3_fold.shape)
                    e_conv3_fold_re= torch.reshape(e_conv3_fold, [512,16])
                    e_conv5 = torch.t(filafter4_w) @ e_conv3_fold_re
                    # print(e_conv5.shape)
                    e_conv5 = torch.reshape(e_conv5, [1, 2304, 16])
                    e_conv5_fold = f.fold(e_conv5, output_size=(4, 4), kernel_size=3, stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv5_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)

                    e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape

                    e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # print('deep', e_conv0.shape)
                    e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv0_fold_unmax,[65536,1])
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model5second(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                        filafter=None,filafter2=None, filafter3=None,filafter4=None,filafter5=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    # no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    # no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    # fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    shape_filafter = filafter.shape
    # print(shape_filafter)
    filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    shape_filafter5 = filafter5.shape
    # print(shape_filafter2)
    filafter5_w = torch.reshape(filafter5,
                                [shape_filafter5[0], shape_filafter5[1] * shape_filafter5[2] * shape_filafter5[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = filafter_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix2 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            # conv_act = fil_w_new @ in_matrix# VX
            # conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            # conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            # conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
            #                                     int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            # in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            conv_act2=filafter_w@in_matrix2
            conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape2 = pool_out2.shape
                pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
                # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # fc_in = torch.reshape(fc_in, [8192,9])
            in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]

            conv_act3= filafter2_w @ in_matrix3


            conv_out3 = fun_front(conv_act3)

            conv_out_shape3 = conv_out3.shape
            conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
                                                  int(math.sqrt(conv_out_shape3[1]))])
            in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            in_matrix6 = create_matrix_x(conv_out5.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act6 = filafter5_w @ in_matrix6
            conv_out6 = fun_front(conv_act6)
            conv_out_shape6 = conv_out6.shape
            conv_flat_shape6 = [conv_out_shape6[0] * conv_out_shape6[1], 1]
            conv_act_flat6 = torch.reshape(conv_act6, conv_flat_shape6)
            conv_out6 = torch.reshape(conv_out6, [1, conv_out_shape6[0], int(math.sqrt(conv_out_shape6[1])),
                                                  int(math.sqrt(conv_out_shape6[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out2, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out6, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter5_w) @ e_conv_out3_reshape

                    e_conv3_shape=e_conv3.shape
                    e_conv3=torch.reshape(e_conv3,[1,e_conv3_shape[0],e_conv3_shape[1]])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_shape = e_conv3_fold.shape
                    # print(e_conv3_fold.shape)
                    e_conv3_fold_re= torch.reshape(e_conv3_fold, [512,16])
                    e_conv5 = torch.t(filafter4_w) @ e_conv3_fold_re
                    # print(e_conv5.shape)
                    e_conv5 = torch.reshape(e_conv5, [1, 2304, 16])
                    e_conv5_fold = f.fold(e_conv5, output_size=(4, 4), kernel_size=3, stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv5_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # print('deep', e_conv4.shape)
                    e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)

                    e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape

                    e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    # e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    # e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # print('deep', e_conv0.shape)
                    # e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    # e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape2)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat2,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape2)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv1_fold_unmax,dot_value.shape)
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape2)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat2 ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix2, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix2)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filafter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_model5fourth(epoch_no, batch_no, in_image, out_image, ker,stri,slope,pool_layer='max',
                         filafter3=None,filafter4=None,filafter5=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # shape_filter = fil.shape

    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    # no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    # no_fil_channels = shape_filter[0]
    # print(no_fil_channels,no_fil_weights)
    # print('fil',fil.shape,'filafter',filafter.shape)
    # fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    # shape_filafter = filafter.shape
    # print(shape_filafter)
    # filafter_w= torch.reshape(filafter, [shape_filafter[0], shape_filafter[1]*shape_filafter[2]*shape_filafter[3]]).to(DEVICE_[0])
    # shape_filafter2 = filafter2.shape
    # print(shape_filafter2)
    # filafter2_w=torch.reshape(filafter2,  [shape_filafter2[0], shape_filafter2[1]*shape_filafter2[2]*shape_filafter2[3]]).to(DEVICE_[0])
    # print(filafter2_w.shape)
    shape_filafter3 = filafter3.shape
    # print(shape_filafter2)
    filafter3_w = torch.reshape(filafter3,
                                [shape_filafter3[0], shape_filafter3[1] * shape_filafter3[2] * shape_filafter3[3]]).to(
        DEVICE_[0])
    shape_filafter4= filafter4.shape
    # print(shape_filafter2)
    filafter4_w = torch.reshape(filafter4,
                                [shape_filafter4[0], shape_filafter4[1] * shape_filafter4[2] * shape_filafter4[3]]).to(
        DEVICE_[0])
    shape_filafter5 = filafter5.shape
    # print(shape_filafter2)
    filafter5_w = torch.reshape(filafter5,
                                [shape_filafter5[0], shape_filafter5[1] * shape_filafter5[2] * shape_filafter5[3]]).to(
        DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])
    # fc_new=filafter_w*fc_w
    # print('fil',fil_w.shape,'filafter_w',filafter_w.shape,'fc_w',fc_w.shape)
    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = filafter3_w
            fc_w_new = fc_w
            # print('in_image[i:i + 1]',in_image[i:i + 1].shape)
            in_matrix4 = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), filafter3, stride, pad)[0]

            fc_out = out_image[i]
            # print('0',filafter_w.shape,fil_w_new.shape ,in_matrix.shape,fc_out.shape)
            # conv_act = fil_w_new @ in_matrix# VX
            # conv_out = fun_front(conv_act)
            # print('convout', conv_out.shape)
            # conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            # conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
            #                                     int(math.sqrt(conv_out_shape[1]))])
            # print('convout', conv_out.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out, pool_ind = f.max_pool2d(conv_out, ker, stri, return_indices=True)
            # print('1',  pool_out.shape)
            # pool_out_shape1 = pool_out.shape
            # pool_flatten_shape1= pool_out_shape1[1] * pool_out_shape1[2] * pool_out_shape1[3] * pool_out_shape1[0]
            # in_matrix2 = create_matrix_x(pool_out.to(DEVICE_[0]), filafter, stride, pad)[0]
            # in_matrix2=torch.reshape(in_matrix2, [576,3*3*64])
            # print('in_matrix2', in_matrix2.shape)
            # conv_act2=filafter_w@in_matrix2
            # conv_out2=fun_front(conv_act2)
            # print('1', filafter_w.shape, pool_out.shape, conv_out2.shape)
            # print(conv_act)
            # print(conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            # conv_out_shape2 = conv_out2.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            # conv_flat_shape2 = [conv_out_shape2[0] * conv_out_shape2[1], 1]
            # conv_act_flat2 = torch.reshape(conv_act2, conv_flat_shape2)
            # print('conv_act_flat2',conv_act_flat2.shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            # conv_out2 = torch.reshape(conv_out2, [1, conv_out_shape2[0], int(math.sqrt(conv_out_shape2[1])),
                                                # int(math.sqrt(conv_out_shape2[1]))])
            # print('conv_out2',conv_out2.shape)
            # Apply     pooling        layer
            # if pool_layer:
            #     if pool_layer == 'avg':
            #         pool_out = f.avg_pool2d(conv_out2, 2, 2)
            #     elif pool_layer == 'max':
            #         #                     print('something')
            #         # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
            #         pool_out2, pool_ind2 = f.max_pool2d(conv_out2, ker, stri, return_indices=True)
            #         # print('pool_ind2',pool_ind2.shape)
            #     pool_out_shape2 = pool_out2.shape
            #     pool_flatten_shape2 = pool_out_shape2[1] * pool_out_shape2[2] * pool_out_shape2[3] * pool_out_shape2[0]
            #     # fc_in = torch.reshape(pool_out2, [pool_flatten_shape2, 1])
            # else:
            #     fc_in = torch.reshape(conv_out2, conv_flat_shape2)
            # # print('2',fc_w_new.shape, filafter_w.shape,pool_out2.shape, fc_in.shape)
            # # fc_in = torch.reshape(fc_in, [8192,9])
            # in_matrix3 = create_matrix_x(pool_out2.to(DEVICE_[0]), filafter2, stride, pad)[0]
            #
            # conv_act3= filafter2_w @ in_matrix3
            #
            #
            # conv_out3 = fun_front(conv_act3)
            #
            # conv_out_shape3 = conv_out3.shape
            # conv_flat_shape3 = [conv_out_shape3[0] * conv_out_shape3[1], 1]
            # conv_act_flat3 = torch.reshape(conv_act3, conv_flat_shape3)
            # conv_out3 = torch.reshape(conv_out3, [1, conv_out_shape3[0], int(math.sqrt(conv_out_shape3[1])),
            #                                       int(math.sqrt(conv_out_shape3[1]))])
            # in_matrix4 = create_matrix_x(conv_out3.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act4 = filafter3_w @ in_matrix4
            conv_out4 = fun_front(conv_act4)
            conv_out_shape4 = conv_out4.shape
            conv_flat_shape4 = [conv_out_shape4[0] * conv_out_shape4[1], 1]
            conv_act_flat4 = torch.reshape(conv_act4, conv_flat_shape4)
            conv_out4 = torch.reshape(conv_out4, [1, conv_out_shape4[0], int(math.sqrt(conv_out_shape4[1])),
                                                  int(math.sqrt(conv_out_shape4[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out4, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out3, pool_ind3 = f.max_pool2d(conv_out4, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape3 = pool_out3.shape
                pool_flatten_shape3 = pool_out_shape3[1] * pool_out_shape3[2] * pool_out_shape3[3] * pool_out_shape3[0]
                # fc_in = torch.reshape(pool_out3, [pool_flatten_shape3, 1])
            else:
                fc_in = torch.reshape(conv_out4, conv_flat_shape4)
            in_matrix5 = create_matrix_x(pool_out3.to(DEVICE_[0]), filafter4, stride, pad)[0]
            conv_act5 = filafter4_w @ in_matrix5
            conv_out5= fun_front(conv_act5)
            conv_out_shape5 = conv_out5.shape
            conv_flat_shape5 = [conv_out_shape5[0] * conv_out_shape5[1], 1]
            conv_act_flat5 = torch.reshape(conv_act5, conv_flat_shape5)
            conv_out5 = torch.reshape(conv_out5, [1, conv_out_shape5[0], int(math.sqrt(conv_out_shape5[1])),
                                                  int(math.sqrt(conv_out_shape5[1]))])
            in_matrix6 = create_matrix_x(conv_out5.to(DEVICE_[0]), filafter3, stride, pad)[0]
            conv_act6 = filafter5_w @ in_matrix6
            conv_out6 = fun_front(conv_act6)
            conv_out_shape6 = conv_out6.shape
            conv_flat_shape6 = [conv_out_shape6[0] * conv_out_shape6[1], 1]
            conv_act_flat6 = torch.reshape(conv_act6, conv_flat_shape6)
            conv_out6 = torch.reshape(conv_out6, [1, conv_out_shape6[0], int(math.sqrt(conv_out_shape6[1])),
                                                  int(math.sqrt(conv_out_shape6[1]))])
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out4, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    # pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out4, pool_ind4 = f.max_pool2d(conv_out6, ker, stri, return_indices=True)
                    # print('pool_ind2',pool_ind2.shape)
                pool_out_shape4 = pool_out4.shape
                pool_flatten_shape4 = pool_out_shape4[1] * pool_out_shape4[2] * pool_out_shape4[3] * pool_out_shape4[0]
                fc_in = torch.reshape(pool_out4, [pool_flatten_shape4, 1])
            else:
                fc_in = torch.reshape(conv_out4, conv_flat_shape4)
            # print(fc_w_new.shape,fc_in.shape)
            y_ = fun_after(fc_w_new @ fc_in)
            # print(y_.shape,fc_out.shape)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer

            # filafter=torch.reshape(filafter,[9,8192])
            # print(filafter_w.shape, torch.t(fc_w_new).shape, e_.shape)
            # filafter = torch.reshape(filafter, [9, 8192])
            e_fc_in =torch.t(fc_w_new) @ e_
            # print('e_fc_in',e_fc_in.shape)
            # e_fc_in2 = torch.t(fc_w_new) @ e_
            if pool_layer:
                # print('back',e_fc_in.shape,pool_out_shape2)
                # print('pool_out_shape3',pool_out_shape3)
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape4)
                # print('e_pool_out', e_pool_out.shape, pool_out_shape2)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    # print('deep', e_conv_out2.shape, e_pool_out.shape)

                    # print(pool_ind2.shape)
                    e_conv_out3 = f.max_unpool2d(e_pool_out, pool_ind4,ker)
                    # print('deep', e_conv_out3.shape, e_pool_out.shape,conv_flat_shape,conv_flat_shape2)
                    # print('deep', e_conv_out3.shape)
                    # print(e_conv_out3.shape)
                    e_conv_out3_reshape = torch.reshape(e_conv_out3, [512,4*4])
                    e_conv3=torch.t(filafter5_w) @ e_conv_out3_reshape

                    e_conv3_shape=e_conv3.shape
                    e_conv3=torch.reshape(e_conv3,[1,e_conv3_shape[0],e_conv3_shape[1]])
                    e_conv3_fold=f.fold(e_conv3,output_size=(4,4),kernel_size=3,stride=1, padding=1)
                    e_conv3_fold_shape = e_conv3_fold.shape
                    # print(e_conv3_fold.shape)
                    e_conv3_fold_re= torch.reshape(e_conv3_fold, [512,16])
                    e_conv5 = torch.t(filafter4_w) @ e_conv3_fold_re
                    # print(e_conv5.shape)
                    e_conv5 = torch.reshape(e_conv5, [1, 2304, 16])
                    e_conv5_fold = f.fold(e_conv5, output_size=(4, 4), kernel_size=3, stride=1, padding=1)

                    e_conv3_fold_unmax=f.max_unpool2d(e_conv5_fold, pool_ind3, ker)
                    # print('deep', e_conv3_fold_unmax.shape)
                    # e_conv3_fold_reshape=torch.reshape(e_conv3_fold_unmax,[256,64])
                    # e_conv4=torch.t(filafter3_w) @ e_conv3_fold_reshape
                    # # print('deep', e_conv4.shape)
                    # e_conv4=torch.reshape(e_conv4,[1,2304,64])
                    # e_conv4_fold=f.fold(e_conv4,output_size=(8,8),kernel_size=3,stride=1, padding=1)
                    # # e_conv3_fold_unmax = f.max_unpool2d(e_conv4_fold, pool_ind3, ker)
                    #
                    # e_conv_out2_reshape = torch.reshape(e_conv4_fold, [256, 64])
                    # e_conv1 = torch.t(filafter2_w) @ e_conv_out2_reshape
                    #
                    # e_conv1 = torch.reshape(e_conv1, [1, 1152, 8 * 8])
                    # e_conv1_fold = f.fold(e_conv1, output_size=(8, 8), kernel_size=3, stride=1, padding=1)
                    # e_conv1_fold_unmax = f.max_unpool2d(e_conv1_fold, pool_ind2, ker)

                    # e_conv1_fold_unmax_reshape=torch.reshape(e_conv1_fold_unmax,[128,16*16])
                    # e_conv0 = torch.t(filafter_w) @ e_conv1_fold_unmax_reshape
                    # print('deep', e_conv0.shape)
                    # e_conv0 = torch.reshape(e_conv0, [1, 576, 256])
                    # e_conv0_fold = f.fold(e_conv0, output_size=(16, 16), kernel_size=3, stride=1, padding=1)
                    # e_conv0_fold_unmax = f.max_unpool2d(e_conv0_fold, pool_ind, ker)

                    # print('e_conv1_fold_unmax',e_conv1_fold_unmax.shape)
                    # e_conv_out3 = f.max_unpool2d(e_conv_out2, pool_ind, ker)
                    # print('deep', e_conv_out3.shape)
                # e_conv_out2 = torch.reshape(e_conv_out2, conv_flat_shape2)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape4)
            # input()
            # print('deri start')
            # t0 = time.time()
###########################
            # e_conv_out2=torch.reshape(e_conv_out2,[128,16*16])
            # # print('backback', torch.t(filafter_w).shape, e_conv_out2.shape)
            # e_fc_in2 = torch.t(filafter_w) @ e_conv_out2
            # print('e_fc_in2',e_fc_in2.shape)

 ######################
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat4,slope)
            # print('dot_value', dot_value.shape)
            # print(fun_front)
            # print(mf.derivative_fun(fun_front))
            # print(conv_act_flat)
            # print(dot_value)
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape4)
            # print('dot_value', dot_value.shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            # print('dot_value',dot_value.shape,e_conv_out2.shape)
            e_conv_out2=torch.reshape(e_conv3_fold_unmax,dot_value.shape)
            e_conv_flat = dot_value * e_conv_out2
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape4)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat4 ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix4, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat4 ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix4, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            # print(fil_w.shape,e_conv.shape,torch.t(in_matrix).shape)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix4)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filafter3)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item(),e_
def inc_solve_2_layer_conv_fc_e2e(epoch_no, in_image, out_image, pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    # print('leeeee', out_image[0])
    out_shape = out_image.shape
    # print(out_image)tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])
    # print(out_image)tensor([[[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.]],
    #
    #         [[0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [0.],
    #          [1.]]], device='cuda:0')
    shape_filter = fil.shape
    # print('fil ', shape_filter) torch.Size([64, 3, 3, 3])
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    # if batch_no + 1 >= 250:
    #     lm = 0.0002
    # if batch_no + 1 >= 375:
    #     lm = 0.0001
    # if batch_no + 1 >= 450:
    #     lm = 0.00005
    # lm = calc_gain(model, in_image)
    # lm = lm/(epoch_no+1)
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        # if batch_no == 0:
        #     print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            # print(i, out_shape[0])
            # t0 = time.time()
            fil_w_new = fil_w
            fc_w_new = fc_w
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
            fc_out = out_image[i]
            conv_act = fil_w_new @ in_matrix# VX
            conv_out = fun_front(conv_act)
            print('conv_act',conv_act)
            print('conv_out', conv_out)
            #             t1 = time.time()
            #             print('create matrix time', t1-t0)

            conv_out_shape = conv_out.shape
            # print('conv_out_shape', conv_out_shape)
            # input()
            # after_flatten_shape = conv_out_shape[0] * conv_out_shape[1]
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            # fc_in = torch.reshape(conv_out, [after_flatten_shape, 1])
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])

            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    #                     print('something')
                    pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                pool_out_shape = pool_out.shape
                pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            else:
                fc_in = torch.reshape(conv_out, conv_flat_shape)
            y_ = fun_after(fc_w_new @ fc_in)
            e_ = fc_out - y_

            #             t2 = time.time()
            #             print('update fc_w time', t2-t1)
            # print("shapes are", fil_w_new.shape, fc_w_new.shape, e_.shape)

            # Backpropagation  to  flattening & pooling   layer
            e_fc_in = torch.t(fc_w_new) @ e_
            if pool_layer:
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, 2)
                e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            # input()
            # print('deri start')
            # t0 = time.time()
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat)
            print(fun_front)
            print(mf.derivative_fun(fun_front))
            # t1 = time.time()
            # print('derivative time', t1-t0)
            # print('haha',conv_flat_shape[0])
            dot_value = dot_value.reshape(conv_flat_shape)
            # e_conv_flat = torch.zeros(conv_flat_shape).to(DEVICE_[0])
            #             print('haha', e_conv_flat.shape, deri_conv_act.shape, e_conv_out.shape)
            #             input()
            e_conv_flat = dot_value * e_conv_out
            #             for k in range(conv_flat_shape[0]):
            #                 e_conv_flat[k] = deri_conv_act[k] * e_conv_out[k]
            # if deri_conv_act[k] != 0:
            #     e_conv_flat[k] = e_fc_in[k]
            # else:
            #     e_conv_flat[k] = 0
            #             t3 = time.time()
            #             print('diagonal time', t3 - t2)
            # print(e_conv_flat.shape, conv_out_shape)
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                # print(fc_w_new.shape, dot_value.shape, conv_act_flat.shape, in_matrix.shape, fil_w_new.shape)
                # print(lm)
                # print(mf.fun_max_derivative(fun_after))
                # print(alpha_w)
                # print(torch.sum(conv_act_flat ** 2))
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                    * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])

                # print(dot_value.shape, dot_value[1000:1200])
                # input()
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                # input()
                lr_con = sum1 - sum2
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, i)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2).to(DEVICE_[0]) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.linalg.eig(lr_con)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)
    #             t4 = time.time()
    #             print('calc time', t4 - t3)
    #             input()

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item()
def inc_train_2_layer(model, train_loader, test_loader, pool_layer='max',start_epoch=0, epochs=400, gain=0.01, auto=True,
                      true_for=1, model_name='_0', avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    print(w1.shape, w2.shape)
    t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        for j in range(start_epoch, epochs):
            t1 = time.time()
            print('============== epoch', j + 1, '/', epochs, '=============')
            gain_rate = gain_schedule(epochs, j)
            gain_ = gain * gain_rate
            if j + 1 > true_for:
                auto = False
            else:
                alpha_vw_min = 1
            gain_adj = gain * alpha_vw_min
            if gain_ > gain_adj:
                gain_ = gain_adj
            for i, (x, y) in enumerate(train_loader):
                if (i + 1) % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                    print('time:', time.time() - t1)
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                pad = curr_layer_front.padding
                stride = curr_layer_front.stride

                w1, w2, alpha_vw, lr = inc_solve_2_layer_conv_fc(j, i, layer_in, layer_tar, pool_layer=pool_layer,
                                                                 fil=w1, fc_wei=w2,
                                                                 fun_front=curr_layer_front.activations,
                                                                 fun_after=curr_layer_after.activations, loop=1,
                                                                 stride=stride, pad=pad, gain=gain_, auto=auto)
                if alpha_vw < alpha_vw_min:
                    alpha_vw_min = alpha_vw
                    print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
                # w = np.matmul(np.linalg.pinv(a), y)

                # if np.remainder(i + 1, 60) == 0:
                #     print("**********", i + 1, "**********")
                #     model.set_weights_index(w1, indx)
                #     model.set_weights_index(w2, -1)
                #     print(model.evaluate_both(train_loader, test_loader))
            #                 if i == 99:
            #                     break
            print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
            model.set_weights_index(w1, indx)
            model.set_weights_index(w2, -1)
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            if j % 5 ==0:
                name1 = ('./saved_models/model7_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
                torch.save(model, name1)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

            if avg_N > 0:
                if k < avg_N - 1:
                    acc_curr_N_epochs += acc_lst[1]/avg_N
                    train_acc_curr_N_epochs += acc_lst[0]/avg_N
                    k += 1
                else:
                    acc_curr_N_epochs += acc_lst[1]/avg_N
                    train_acc_curr_N_epochs += acc_lst[0]/avg_N
                    k = 0
                    if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
                        print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                        print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
                        acc_last_N_epochs = acc_curr_N_epochs
                        acc_curr_N_epochs = 0
                        if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
                            print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
                            print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
                            break
                        print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
                        train_acc_last_N_epochs = train_acc_curr_N_epochs
                        train_acc_curr_N_epochs = 0
                    else:
                        print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                        print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
                        acc_last_N_epochs = acc_curr_N_epochs
                        acc_curr_N_epochs = 0
                        print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
                        train_acc_last_N_epochs = train_acc_curr_N_epochs
                        train_acc_curr_N_epochs = 0

            print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    print('time:', time.time() - t0)
    return w1, w2

def inc_train_2_layer_e2e(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    # print(indx)
    w1 = model.get_weights_index(indx)



    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        w1, w2, alpha_vw, lr, e= inc_solve_2_layer_conv_fc(j, i,layer_in, layer_tar,ker,stri,0,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2
def inc_train_2_layer_e2e_acce(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        indx = -4
    else:
        indx = -3
    # model=model.to(DEVICE_[0])
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    with torch.no_grad():
        # max_acc_test = 0
        # j_at_max = 0
        # j_max_old = 0
        # alpha_vw_min = 1
        # acc_last_N_epochs = 0.
        # acc_curr_N_epochs = 0.
        # train_acc_last_N_epochs = 0.
        # train_acc_curr_N_epochs = 0.
        # k = 0
        # # for j in range(epochs):
        # # for j in range(start_epoch, epochs):
        # t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        # if j + 1 > true_for:
        #     auto = False
        # else:
        #     alpha_vw_min = 1
        # gain_adj = gain * alpha_vw_min
        # if gain_ > gain_adj:
        #     gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        # t0 = time.time()
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # pad = curr_layer_front.padding
        # print(pad)
        # stride = curr_layer_front.stride
        # print(stride)
        w1, w2, alpha_vw, lr, e= inc_solve_2_layer_conv_fc_acce(j, i,layer_in, layer_tar,ker,stri,0,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=1, pad=1, gain=gain_, auto=auto)
        # if alpha_vw < alpha_vw_min:
        #     alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2
def inc_train_2_layer_e2e_DA(model, i,j,x,y,xs,ys,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        layer_in_source = model.forward_to_layer(xs.float().to(DEVICE_[0]), indx)
        layer_tar_source = one_hot_embedding(ys.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        w1, w2, alpha_vw, lr, e= inc_solve_2_layer_conv_fc(j, i,layer_in, layer_tar,ker,stri,0,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        w1_source, w2_source, alpha_vw, lr, e = inc_solve_2_layer_conv_fc(j, i, layer_in_source, layer_tar_source, ker, stri, 0,
                                                            pool_layer=pool_layer,
                                                            fil=w1, fc_wei=w2,
                                                            fun_front=curr_layer_front.activations,
                                                            fun_after=curr_layer_after.activations, loop=1,
                                                            stride=stride, pad=pad, gain=gain_, auto=auto)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(0.9*w1+0.1*w1_source, indx)
        model.set_weights_index(0.9*w2+0.1*w2_source, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2
def inc_train_2_layer_e2ev(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        # print(layer_tar.shape)
        w1, w2, alpha_vw, lr ,err= inc_solve_2_layer_conv_fc_batch(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        error=0
        for i in range(len(err)):
            # print(i)
            error = error + err[i] * err[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,error

def inc_train_2_layer_e2evinverse(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = -3
    # print(indx)
    w1 = model.get_weights_index(0)
    wnext1 = model.get_weights_index(2)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        # print(layer_tar.shape)
        w1, w2, alpha_vw, lr ,err= inc_solve_2_layer_conv_fc_inverse(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1, filnext=wnext1,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        error=0
        for i in range(len(err)):
            # print(i)
            error = error + err[i] * err[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,error
def inc_train_2_layer_e2efirst(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_first(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err

def inc_train_2_layer_e2emodel2first(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+4)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model2first(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,filafter2=w1afterafter, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel2deep(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 2
    else:
        indx =2
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    # w1afterafter=model.get_weights_index(indx+4)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model2deep(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel3first(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+4)
    wconv3=model.get_weights_index(indx+5)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model3first(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,filafter2=w1afterafter, filafter3=wconv3,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel3second(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 2
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+3)
    # wconv3=model.get_weights_index(indx+5)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model3second(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                        filafter=w1,filafter2=w1after,filafter3=w1afterafter,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel3deep(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 4
    else:
        indx = 4
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+1)
    # w1afterafter=model.get_weights_index(indx+4)
    # wconv3=model.get_weights_index(indx+5)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model3deep(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,fc_wei=w2, fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel4deep(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx =5
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    # w1afterafter=model.get_weights_index(indx+4)
    # wconv3=model.get_weights_index(indx+5)
    # wconv4 = model.get_weights_index(indx + 7)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model4deep(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel5deep(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx =7
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+1)
    # w1afterafter=model.get_weights_index(indx+4)
    # wconv3=model.get_weights_index(indx+5)
    # wconv4 = model.get_weights_index(indx + 7)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model5deep(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel4first(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+4)
    wconv3=model.get_weights_index(indx+5)
    wconv4 = model.get_weights_index(indx + 7)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model4first(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,filafter2=w1afterafter, filafter3=wconv3, filafter4=wconv4,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel4second(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 2
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+3)
    wconv3=model.get_weights_index(indx+5)
    # wconv4 = model.get_weights_index(indx + 7)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model4second(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         filafter=w1,filafter2=w1after, filafter3=w1afterafter, filafter4=wconv3,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel4third(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 4
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+1)
    w1afterafter=model.get_weights_index(indx+3)
    # wconv3=model.get_weights_index(indx+5)
    # wconv4 = model.get_weights_index(indx + 7)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model4third(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                                           filafter2=w1, filafter3=w1after,
                                                                           filafter4=w1afterafter,fc_wei=w2,
                                                                           fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel5first(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 0
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+4)
    wconv3=model.get_weights_index(indx+5)
    wconv4 = model.get_weights_index(indx + 7)
    wconv5 = model.get_weights_index(indx +8)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model5first(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                         fil=w1,filafter=w1after,filafter2=w1afterafter, filafter3=wconv3, filafter4=wconv4,filafter5=wconv5,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel5second(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 2
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+3)
    wconv3=model.get_weights_index(indx+5)
    wconv4 = model.get_weights_index(indx + 6)
    # wconv5 = model.get_weights_index(indx +8)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model5second(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                       filafter=w1,filafter2=w1after, filafter3=w1afterafter, filafter4=wconv3,filafter5=wconv4,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2emodel5fourth(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = 5
    else:
        indx = 0
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w1after = model.get_weights_index(indx+2)
    w1afterafter=model.get_weights_index(indx+3)
    # wconv3=model.get_weights_index(indx+5)
    # wconv4 = model.get_weights_index(indx + 6)
    # wconv5 = model.get_weights_index(indx +8)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)

        w1, w2, alpha_vw, lr ,error= inc_solve_2_layer_conv_fc_model5fourth(j, i,layer_in, layer_tar,ker,stri,curr_layer_front.slope,pool_layer=pool_layer,
                                                      filafter3=w1, filafter4=w1after, filafter5=w1afterafter,fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        # model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2,err
def inc_train_2_layer_e2ew(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        w1, w2, alpha_vw, lr,error = inc_solve_2_layer_conv_fc(j, i,layer_in, layer_tar,ker,stri,0.01,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
    #     model.set_weights_index(w1, indx)
        model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)

    return w1, w2,err
def inc_train_2_layer_e2ewbn(model, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    if pool_layer:
        # print(pool_layer, 'pooling')
        indx = -4-1
    else:
        indx = -3-1
    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    curr_layer_frontnext = model.layers[indx+1]
    # print(curr_layer_frontnext.activations)
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        w1, w2, alpha_vw, lr,error = inc_solve_2_layer_conv_fc_bn_batch(j, i,layer_in, layer_tar,ker,stri,0.01,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_frontnext.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        err = 0

        for i in range(len(error)):
            # print(i)
            err = err + error[i] * error[i]
        # print(err)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
    #     model.set_weights_index(w1, indx)
        model.set_weights_index(w2, -1)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)

    return w1, w2,err

# Train any conv and fc layer we like (choose own starting index)
def inc_train_2_layer_e2e_INDEX(model,indx, i,j,x,y,ker,stri,pool_layer='max', epochs=400, gain=0.001, auto=True,
                      true_for=1, avg_N = 0):
    # weights = model.get_weights_2()
    # if pool_layer:
    #     # print(pool_layer, 'pooling')
    #     indx = -4
    # else:
    #     indx = -3

    # print(indx)
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(indx+3)
    curr_layer_after = model.layers[indx+3]
    # print(w1.shape, w2.shape)
    # t0 = time.time()
    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        # for j in range(epochs):
        # for j in range(start_epoch, epochs):
        t1 = time.time()
        # print('============== epoch', j + 1, '/', epochs, '=============')
        gain_rate = gain_schedule(epochs, j)
        gain_ = gain * gain_rate
        if j + 1 > true_for:
            auto = False
        else:
            alpha_vw_min = 1
        gain_adj = gain * alpha_vw_min
        if gain_ > gain_adj:
            gain_ = gain_adj
            # for i, (x, y) in enumerate(train_loader):
        # if (i + 1) % (len(train_loader) // 10) == 0:
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        #     print('time:', time.time() - t1)
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        pad = curr_layer_front.padding
        stride = curr_layer_front.stride
        # print(curr_layer_front.activations)
        # print(curr_layer_after.activations)
        w1, w2, alpha_vw, lr = inc_solve_2_layer_conv_fc(j, i,layer_in, layer_tar,ker,stri,pool_layer=pool_layer,
                                                         fil=w1, fc_wei=w2,
                                                         fun_front=curr_layer_front.activations,
                                                         fun_after=curr_layer_after.activations, loop=1,
                                                         stride=stride, pad=pad, gain=gain_, auto=auto)
        if alpha_vw < alpha_vw_min:
            alpha_vw_min = alpha_vw
            # print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)
        # w = np.matmul(np.linalg.pinv(a), y)

        # if np.remainder(i + 1, 60) == 0:
        #     print("**********", i + 1, "**********")
        #     model.set_weights_index(w1, indx)
        #     model.set_weights_index(w2, -1)
        #     print(model.evaluate_both(train_loader, test_loader))
    #                 if i == 99:
    #                     break
    #     print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
        model.set_weights_index(w1, indx)
        model.set_weights_index(w2, indx+3)
        # if i % 468 == 0:
        #     acc_lst = model.evaluate_both_e2e(modelnum,train_loader, test_loader)
        #     print(acc_lst)
        # if j+1 % 5 ==0:
        #     name1 = ('./saved_models_e2e/model'+'%d' % modelnum+'_kmnist' + '_%f_' % acc_lst[1] + '%d.pkl' % j)
        #     torch.save(model, name1)
        # # save model
        # if model_name != '_0':
        #     if acc_lst[1] > max_acc_test:
        #         max_acc_test = acc_lst[1]
        #         j_max_old = j_at_max
        #         j_at_max = j
        #     model.save_current_state_e2e(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
        #
        # if avg_N > 0:
        #     if k < avg_N - 1:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k += 1
        #     else:
        #         acc_curr_N_epochs += acc_lst[1]/avg_N
        #         train_acc_curr_N_epochs += acc_lst[0]/avg_N
        #         k = 0
        #         if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
        #                 print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
        #                 print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
        #                 # break
        #             print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0
        #         else:
        #             print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
        #             print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
        #             acc_last_N_epochs = acc_curr_N_epochs
        #             acc_curr_N_epochs = 0
        #             print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
        #             train_acc_last_N_epochs = train_acc_curr_N_epochs
        #             train_acc_curr_N_epochs = 0

        # print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    # print('time:', time.time() - t0)
    return w1, w2

# learn convolutional filter weights progressively. reduce error to get the weights that can produce output from input
def inc_solve_filter(batch, lin, in_images, out_images, fil, func, loop, ran_mix, gain_rate, pad):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)

        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)

        # out_images is post activation for non-linear activation and is preactivation value for linear activation
        out_images = mf.fun_cut(out_images, func)

    in_shape = in_images.shape
    out_shape = out_images.shape

    out_images = torch.reshape(out_images, [out_shape[0], out_shape[1], out_shape[2] * out_shape[3]])

    shape_filter = fil.shape
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights])

    matrix_x = create_matrix_x(in_images, fil,1, pad)

    gain = 1e-6  # old: 1e-4, after resizing to (224,224)--> 1e-5
    lr = gain  # * gain_rate

    if ran_mix:
        print('inc_solve_x_random_shuffle')
        matrix_x, out_images = data_randomize(matrix_x, out_images)

    for j in range(loop):

        if loop > 1:
            if j == math.ceil(loop / 2) + 1:
                lr = lr / 2
            if j == math.ceil(3 * loop / 4) + 1 & loop > 4:
                lr = lr / 2
            if j == loop - 1 & loop > 5:
                lr = lr / 5
            if j == loop & loop > 8:
                lr = lr / 10
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(in_shape[0]):
            w_new = fil_w
            in_matrix = matrix_x[k, :, :]
            y_ = w_new @ in_matrix
            if ~lin:
                # print('nonlinear coming here')
                y_ = func(y_)
            e_ = torch.squeeze(out_images[k, :, :]) - y_
            fil_w = w_new + lr * e_ @ torch.t(in_matrix)

    x = torch.reshape(fil_w, shape_filter)

    return x

# update fully connected layer weights
def inc_solve_fc(learnrate,batch, lin, in_images, out_images, weight, func, loop, ranmix, gain_, gain_rate):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)
        out_images = mf.fun_cut(out_images, func)

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    # if gain_ < 0:
    #     #         t0 = time.time()
    #     # print(torch.t(in_images[0, :]).device)
    #     max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
    #     #         # t05 = time.time()
    #     #         # print('max0 time:', t05 - t0, max_phi2)
    #     #         # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
    #     #         t1 = time.time()
    #     #         print('max time', t1 - t0, max_phi2)
    #     #     gain = 1 / max_phi2 * gain_rate  # * torch.ones(out_shape[1], 1)
    #     gain = 1 / max_phi2 * gain_rate
    # else:
    #     gain = gain_ * gain_rate
    # lr = gain  # .to(DEVICE_[0])

    if ranmix:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    for j in range(loop):
        # print('number of loop:', loop)
        # lr = lr * gain_schedule(loop, j)
        # print('lr',lr)
        lr=learnrate
        # print('lr',lr)
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
                # print('maxphi', max_phi2, gain_rate)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(out_shape[0]):
            # print('weight',weight)
            w_new=torch.stack(list(weight), dim=0)
            # w_new = weight
            in_matrix = in_images[k:k + 1, :]
            # print(in_matrix.device)
            # print('w_new',w_new)
            # print('in_iamges',in_images.type)
            y_ = w_new @ torch.t(in_matrix)
            if ~lin:
                # print('nonlinear coming here')
                y_ = func(y_)
            #             print(torch.t(out_images[k:k + 1, :]).shape, y_.shape)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, in_matrix.shape)
            e_phi = e_ @ in_matrix
            # print(lr.shape, e_phi.shape)
            weight = w_new + lr * e_phi
            # print(weight.device)
    # print('weight',weight)
    return weight,e_
    # return weight, lr.item()

# train last conv layer weights
def inc_train_filter(model, train_loader, test_loader, epoch=2, loop=10, ran_mix=False):
    weights = model.get_weights()
    w = weights[-1]
    print(model.evaluate_both(train_loader, test_loader))

    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                print('=========== batch', i + 1, '/', len(train_loader), '==========')
                w = inc_solve_filter(i, 0, x.float(), y.float(), w, f.relu, loop, ran_mix, gain_rate)
                # w = np.matmul(np.linalg.pinv(a), y)
                # out = model.predict(x_train)
                # print('laalalaa')
                # print(out - y_test)
            # if np.remainder(j + 1, 100) == 0:
            #     print("**********", j + 1, "**********")
            #     model.set_weights_2([w])
            #     print(evaluate_both(model, train_loader, test_loader))
            # print(w[3, :, :, :])
            # print(w[4, :, :, :])
            model.set_weights([w])
            print(model.evaluate_both(train_loader, test_loader))

    return w

# train one layer (at_layer) whether it is conv or fc layer
def inc_train_1_layer(model, at_layer, train_loader, test_loader, epoch=20, loop=1, gain_=0.1,
                      ran_mix=False, model_name='_0'):
    curr_layer = model.layers[at_layer]
    w = model.get_weights_index(at_layer)
    # print(model.evaluate_both(train_loader, test_loader))

    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule_old(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                if i % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                #                 print(model.no_layers - at_layer - 1)
                #                 input()
                layer_tar = model.backward_to_layer(layer_tar, model.no_layers - at_layer - 1)
                if curr_layer.name == 'conv':
                    pad = curr_layer.padding
                    w = inc_solve_filter(i, False, layer_in, layer_tar,
                                         w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                elif curr_layer.name == 'fc':
                    w, lr = inc_solve_fc(i, False, layer_in, layer_tar,
                                         w, curr_layer.activations, loop, ran_mix, gain_, gain_rate)

            model.set_weights_index(w, at_layer)
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 0)

    return w


def inverse_layerwise_training(model, train_loader, test_loader,
                               config, no_layers=1, epoch=1, loop=1, gain_=-1, mix_data=False, model_name='_0'):
    for i in range(no_layers):
        cur_inx = model.no_layers - i - 1
        # print('cur_inx is:', cur_inx)
        cur_lay = model.layers[cur_inx]
        print('First time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer(model, cur_inx, train_loader, test_loader, epoch, loop, gain_,
                                           mix_data, model_name)
            model.set_weights_index(out_weight, cur_inx)
    for i in range(no_layers - 1):
        cur_inx = i + model.no_layers - no_layers + 1
        cur_lay = model.layers[cur_inx]
        print('Second time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer(model, cur_inx, train_loader, test_loader, epoch, loop, gain_,
                                           mix_data, model_name)
            model.set_weights_index(out_weight, cur_inx)
    return 0

# update a 2 layer (1 conv and 1fc) network weight
def inc_solve_2_fc_layer(model, train_loader, test_loader, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix = in_images[k:k + 1, :]
            vx_ = v_new @ torch.t(in_matrix)
            phi = fun1(vx_)
            if ~lin:
                # print('nonlinear coming here')
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_)
            dot_a_vx_ = torch.squeeze(dot_a_vx)

            if auto:

                #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
                #             # check lr
                #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
                #             print(in_matrix.shape, w_new.shape)
                #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                lr_con = torch.diagflat(
                    (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2) * torch.ones(
                        out_shape[1], 1)).to(
                    DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                    w_new) * lr
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, k)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    print(alpha_v)
                    lr_con = torch.diagflat(
                        (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2) * torch.ones(
                            out_shape[1], 1)).to(
                        DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                        w_new) * lr
                    eig_values, _ = torch.linalg.eig(lr_con)
            #                 print(eig_values[:, 0])
            # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item()

def backupinc_solve_2_fc_layer_e2e(model, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix =in_images[k:k + 1, :]
            # print(v_new.shape,torch.t(in_matrix).shape)

            vx_ = v_new @ torch.t(in_matrix)
            # print('vx_',vx_.shape)
            phi = fun1(vx_)
            if ~lin:
                # print('nonlinear coming here')
                # print('wa',w_new.shape,phi.shape)
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi
            # print(y_.shape, out_images[k:k + 1, :].shape)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_,slope)
            dot_a_vx_ = torch.squeeze(dot_a_vx)

            if auto:

                #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
                #             # check lr
                #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
                #             print(in_matrix.shape, w_new.shape)
                #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                lr_con = torch.diagflat(
                    (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
                        out_shape[1], 1).to(DEVICE_[0]) ).to(
                    DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                    w_new).to(DEVICE_[0])  * lr
                eig_values, _ = torch.linalg.eig(lr_con)
                # print(eig_values.shape)
            #     while (eig_values[:] < -0.005).any():
            #         # print('%d - %d', j, k)
            #         #                 print(eig_values[:, 0])
            #         alpha_v = alpha_v / 1.1
            #         alpha_w = alpha_w / 1.1
            #         # print(alpha_v)
            #         lr_con = torch.diagflat(
            #             (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #                 out_shape[1], 1).to(DEVICE_[0]) ).to(
            #             DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #             w_new).to(DEVICE_[0])  * lr
            #         eig_values, _ = torch.linalg.eig(lr_con)
            # #                 print(eig_values[:, 0])
            # # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item(),e_
def inc_solve_2_fc_layer_e2e_deep(model, batch, lin, in_images_first,in_images, out_images,
                                  weight_v_first, weight_v, weight_w, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            v_first_new= weight_v_first
            in_matrix =in_images[k:k + 1, :]
            in_matrix_first = in_images_first[k:k + 1, :]
            # print(v_new.shape,torch.t(in_matrix).shape)

            vx_ = v_new @ torch.t(in_matrix)
            vx_first_ = v_first_new @ torch.t(in_matrix_first)
            # print('vx_',vx_.shape)
            phi = fun1(vx_)
            # print('phi',phi.shape)
            phi_=torch.squeeze(phi)
            # print('phi', phi_)
            R=torch.diag(phi_)
            # print('R', R.shape)
            R = torch.tensor(R).to(DEVICE_[0])

            if ~lin:
                # print('nonlinear coming here')
                # print('wa',w_new.shape,phi.shape)
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi
            # print(y_.shape, out_images[k:k + 1, :].shape)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_,slope)
            dot_a_vx_ = torch.squeeze(dot_a_vx)
            # print('dot_a_vx_',dot_a_vx_.shape)

            dot_a_vx_first = dot_f1(vx_first_, slope)
            dot_a_vx_first_ = torch.squeeze(dot_a_vx_first)
            # print('dot_a_vx_first', dot_a_vx_first.shape)
            if auto:

                #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
                #             # check lr
                #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
                #             print(in_matrix.shape, w_new.shape)
                #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                lr_con = torch.diagflat(
                    (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
                        out_shape[1], 1).to(DEVICE_[0]) ).to(
                    DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                    w_new).to(DEVICE_[0])  * lr
                eig_values, _ = torch.linalg.eig(lr_con)
                # print(eig_values.shape)
            #     while (eig_values[:] < -0.005).any():
            #         # print('%d - %d', j, k)
            #         #                 print(eig_values[:, 0])
            #         alpha_v = alpha_v / 1.1
            #         alpha_w = alpha_w / 1.1
            #         # print(alpha_v)
            #         lr_con = torch.diagflat(
            #             (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #                 out_shape[1], 1).to(DEVICE_[0]) ).to(
            #             DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #             w_new).to(DEVICE_[0])  * lr
            #         eig_values, _ = torch.linalg.eig(lr_con)
            # #                 print(eig_values[:, 0])
            # # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix
            # print(weight_v_first.shape,v_first_new.shape,weight_v.shape,weight_w.shape)
            # print(dot_a_vx.shape,dot_a_vx_first.shape)
            # print(((dot_a_vx * (torch.t(w_new ) @ e_))).shape, in_matrix.shape)
            # print(((dot_a_vx_first * (torch.t(w_new@v_new) @ e_)) ).shape,  in_matrix_first.shape)
            weight_v_first = v_first_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx_first * (torch.t(w_new@R@v_new) @ e_)) @ in_matrix_first
        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item()

# batch update in_matrix = in_images(batch instead of single sample)
def batchinc_solve_2_fc_layer_e2e(model, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix =in_images
            # print(v_new.shape,torch.t(in_matrix).shape)

            vx_ = v_new @ torch.t(in_matrix)
            # print('vx_',vx_.shape)
            phi = fun1(vx_)
            if ~lin:
                # print('nonlinear coming here')
                # print('wa',w_new.shape,phi.shape)
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi
            e_ = torch.t(out_images ) - y_
            print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_,slope)
            dot_a_vx_ = torch.squeeze(dot_a_vx)

            # if auto:
            #
            #     #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
            #     #             # check lr
            #     #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
            #     #             print(in_matrix.shape, w_new.shape)
            #     #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
            #     #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
            #     lr_con = torch.diagflat(
            #         (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #             out_shape[1], 1).to(DEVICE_[0]) ).to(
            #         DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #         w_new).to(DEVICE_[0])  * lr
            #     eig_values, _ = torch.linalg.eig(lr_con)
            #     while (eig_values[:, 0] < -0.005).any():
            #         # print('%d - %d', j, k)
            #         #                 print(eig_values[:, 0])
            #         alpha_v = alpha_v / 1.1
            #         alpha_w = alpha_w / 1.1
            #         # print(alpha_v)
            #         lr_con = torch.diagflat(
            #             (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #                 out_shape[1], 1).to(DEVICE_[0]) ).to(
            #             DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #             w_new).to(DEVICE_[0])  * lr
            #         eig_values, _ = torch.linalg.eig(lr_con)
            #                 print(eig_values[:, 0])
            # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item()

# has bn flag to control whether to add bn layer between first conv and activation function; batch update in_matrix = in_images(batch instead of single sample)
def inc_solve_2_fc_layer_e2e(bnflag,model, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix = in_images
            # print(v_new.shape,torch.t(in_matrix).shape)

            vx_ = v_new @ torch.t(in_matrix)
            # print('vx_',vx_.shape)
            if bnflag==1:
             # print(bnflag)
             bn= nn.BatchNorm1d(512).to(DEVICE_[0])
             vx_=bn(torch.t(vx_))
             vx_=torch.t(vx_)
            phi = fun1(vx_)
            # print(fun1)
            # print('phi', phi.shape,'w_new',w_new.shape)
            if ~lin:
                # print('nonlinear coming here')
                # print('wa',w_new.shape,phi.shape)
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi

            e_ = torch.t(out_images) - y_
            # print(e_.shape, phi.shape)
            # print('y_', y_.shape, 'e_', e_.shape)
            dot_f1 = mf.derivative_fun(fun1)
            # print('vx_', vx_.shape)
            dot_a_vx = dot_f1(vx_,slope)
            dot_a_vx_ = torch.squeeze(dot_a_vx)
            # print('dot_a_vx_', dot_a_vx_.shape)
            # if auto:
            #
            #     #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
            #     #             # check lr
            #     #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
            #     #             print(in_matrix.shape, w_new.shape)
            #     #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
            #     #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
            #     # print(in_matrix.shape,w_new.shape,dot_a_vx_.shape)
            #     # torch.sum(in_matrix ** 2).to(DEVICE_[0]) * lr * w_new * dot_a_vx_ ** 2
            #     lr_con = torch.diagflat(
            #         (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #             out_shape[1], 1).to(DEVICE_[0]) ).to(
            #         DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #         w_new).to(DEVICE_[0])  * lr
            #     eig_values, _ = torch.linalg.eig(lr_con)
            #     while (eig_values[:, 0] < -0.005).any():
            #         # print('%d - %d', j, k)
            #         #                 print(eig_values[:, 0])
            #         alpha_v = alpha_v / 1.1
            #         alpha_w = alpha_w / 1.1
            #         # print(alpha_v)
            #
            #         lr_con = torch.diagflat(
            #             (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
            #                 out_shape[1], 1).to(DEVICE_[0]) ).to(
            #             DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
            #             w_new).to(DEVICE_[0])  * lr
            #         eig_values, _ = torch.linalg.eig(lr_con)
            #                 print(eig_values[:, 0])
            # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix
            # print(weight_v)
        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item(),e_
def inc_solve_2_fc_layer_e2ebnbatch(bnflag,model, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
    # for k in range(out_shape[0]):
        v_new = weight_v
        w_new = weight_w
        in_matrix =in_images
        # print(v_new.shape,torch.t(in_matrix).shape)

        vx_ = v_new @ torch.t(in_matrix)
        vx_batch = v_new @ torch.t(in_images)
        # print('vx_',vx_.shape)
        if bnflag==1:
         # print(bnflag)
         bn= nn.BatchNorm1d(512).to(DEVICE_[0])
         vx_batch=bn(torch.t(vx_batch))
         vx_batch=torch.t(vx_batch)
         phi = fun1(vx_batch )
        phi = fun1(vx_)
        # print(fun1)
        # print('phi', phi.shape,'w_new',w_new.shape)
        if ~lin:
            # print('nonlinear coming here')
            # print('wa',w_new.shape,phi.shape)
            wa = w_new @ phi
            y_ = fun2(wa)
        else:
            y_ = w_new @ phi

        e_ = torch.t(out_images ) - y_
        # print(e_.shape, phi.shape)
        # print('y_', y_.shape, 'e_', e_.shape)
        dot_f1 = mf.derivative_fun(fun1)
        # print('vx_', vx_.shape)
        dot_a_vx = dot_f1(vx_,slope)
        dot_a_vx_ = torch.squeeze(dot_a_vx)
        # print('dot_a_vx_', dot_a_vx_.shape)
        # if auto:
        #
        #     #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
        #     #             # check lr
        #     #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
        #     #             print(in_matrix.shape, w_new.shape)
        #     #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
        #     #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
        #     # print(in_matrix.shape,w_new.shape,dot_a_vx_.shape)
        #     # torch.sum(in_matrix ** 2).to(DEVICE_[0]) * lr * w_new * dot_a_vx_ ** 2
        #     lr_con = torch.diagflat(
        #         (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
        #             out_shape[1], 1).to(DEVICE_[0]) ).to(
        #         DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
        #         w_new).to(DEVICE_[0])  * lr
        #     eig_values, _ = torch.linalg.eig(lr_con)
        #     while (eig_values[:, 0] < -0.005).any():
        #         # print('%d - %d', j, k)
        #         #                 print(eig_values[:, 0])
        #         alpha_v = alpha_v / 1.1
        #         alpha_w = alpha_w / 1.1
        #         # print(alpha_v)
        #
        #         lr_con = torch.diagflat(
        #             (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
        #                 out_shape[1], 1).to(DEVICE_[0]) ).to(
        #             DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
        #             w_new).to(DEVICE_[0])  * lr
        #         eig_values, _ = torch.linalg.eig(lr_con)
        #                 print(eig_values[:, 0])
        # time.sleep(0.5)

        if batch == 0  :
            if loop <= 20:
                print(['loop ', j + 1])
                print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

        weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
        weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                   (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix
        # print(weight_v)
        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item()

def inc_solve_deep_fc_layer_e2e(model, batch, lin, in_images, out_images,
                         weight_v, weight_w,weight_u, fun1, fun2, loop, mix_data, slope,gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    # print(in_images[0].shape)
    # print(in_images[0] * in_images[0])
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        # t05 = time.time()
        # print('max0 time:', t05 - t0, max_phi2)
        # max_phi2 = max(torch.diagonal(in_images @ torch.t(in_images)))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    # t2 = time.time()
    # print('ranmix time:', t2-t1)
    lr_total = gain_
    for j in range(loop):
        # print('number of loop:', loop)
        lr = gain_

        # print(in_images.shape, out_images.shape, weight_v.shape, weight_w.shape)
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix = in_images[k:k + 1, :]
            vx_ = v_new @ torch.t(in_matrix)
            phi = fun1(vx_)
            if ~lin:
                # print('nonlinear coming here')
                wa = weight_u @ phi
                wa_= fun1(wa)
                y_ = fun2(w_new @ wa_)
            else:
                y_ = w_new @ phi
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_,slope)
            dot_a_vx_ = torch.squeeze(dot_a_vx)

            if auto:

                #             print((w_new * torch.squeeze(dot_a_vx)**2).shape, w_new.shape) # dot_a_vx_**2
                #             # check lr
                #             print('first value is', torch.diagflat((2 * lr - alpha_w * torch.sum(phi ** 2)* lr **2)* torch.ones(out_shape[1], 1)))
                #             print(in_matrix.shape, w_new.shape)
                #             print('second value is', alpha_v * torch.sum( in_matrix ** 2) * lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                #             print('third is', lr * w_new * dot_a_vx_**2 @ torch.t(w_new) * lr)
                lr_con = torch.diagflat(
                    (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
                        out_shape[1], 1).to(DEVICE_[0]) ).to(
                    DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                    w_new).to(DEVICE_[0])  * lr
                eig_values, _ = torch.linalg.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    # print('%d - %d', j, k)
                    #                 print(eig_values[:, 0])
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    # print(alpha_v)
                    lr_con = torch.diagflat(
                        (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2).to(DEVICE_[0])  * torch.ones(
                            out_shape[1], 1).to(DEVICE_[0]) ).to(
                        DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2).to(DEVICE_[0])  * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                        w_new).to(DEVICE_[0])  * lr
                    eig_values, _ = torch.linalg.eig(lr_con)
            #                 print(eig_values[:, 0])
            # time.sleep(0.5)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    # print('deep',weight_v.shape,weight_w.shape)
    return weight_v, weight_w, alpha_w, lr_total.item()

# update the weight of last 2 layers of the network
def conv_train_2_fc_layer_last(model, train_loader, test_loader, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)
    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-2)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
            for i, (x, y) in enumerate(train_loader):
                print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -2)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                # print(layer_tar[0:14])
                # input()
                v, w, alpha_w_, lr = inc_solve_2_fc_layer(model, train_loader, test_loader, i, False, layer_in, layer_tar,
                                                      v, w, curr_layer_front.activations, curr_layer_after.activations,
                                                      loop, ran_mix, gain_rate=gain_rate, gain_=gain_, auto=auto)
            #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
            model.set_weights_index(w, -1)
            model.set_weights_index(v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w

def batchconv_train_2_fc_layer_last_e2e(model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-2)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -2)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr = batchinc_solve_2_fc_layer_e2e(model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        model.set_weights_index(w, -1)
        model.set_weights_index(v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w
def bnconv_train_2_fc_layer_last_e2e(bnflag,model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    # batch normalization layer to get activation
    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-3)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr = inc_solve_2_fc_layer_e2e(bnflag,model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        model.set_weights_index(w, -1)
        model.set_weights_index(v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w
# update v only
def bnconv_train_2_fc_layer_last_e2ev(bnflag,model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-3)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr,err = inc_solve_2_fc_layer_e2e(bnflag,model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        # model.set_weights_index(w, -1)
        model.set_weights_index(v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w

# update w only 
def  bnconv_train_2_fc_layer_last_e2ew(bnflag,model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-3)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr,err = inc_solve_2_fc_layer_e2e(bnflag,model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        error=0
        ferror=0
        for i in range(len(err)):
            # print(i)
            error = error + err[i] * err[i]
        for i in range(len(error)):
            # print(i)
            ferror = ferror + error[i] * error[i]
        ferror= ferror/128
        # print(ferror)
        # print(error.shape)
        # print(err.shape)
        model.set_weights_index(w, -1)
        # model.set_weights_index(v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w,ferror

# update last conv and fc layer weights (-3 and -1; -2 is flatten)
def  conv_train_2_fc_layer_last_e2e(model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    curr_layer_front = model.layers[-3]
    v = model.get_weights_index(-3)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr,err = backupinc_solve_2_fc_layer_e2e(model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        error = 0
        for i in range(len(err)):
            # print(i)
            error = error + err[i] * err[i]
        # print(err)
        model.set_weights_index(w, -1)
        model.set_weights_index(v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w,error

# blend results from 2 solvers (2 samples)
def  conv_train_2_fc_layer_last_e2e_DA(model, i,j,x,y,xs,ys, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)

    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-2)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -2)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()
        layer_in_source = model.forward_to_layer(xs.float().to(DEVICE_[0]), -2)
        layer_tar_source = one_hot_embedding(ys.long(), model.no_outputs).to(DEVICE_[0]).float()
        v, w, alpha_w_, lr = backupinc_solve_2_fc_layer_e2e(model, i, False, layer_in, layer_tar,
                                              v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        v_source, w_source, alpha_w_, lr = backupinc_solve_2_fc_layer_e2e(model, i, False, layer_in_source, layer_tar_source,
                                                            v, w, curr_layer_front.activations,
                                                            curr_layer_after.activations,
                                                            loop, ran_mix, 0.01, gain_rate=gain_rate, gain_=gain_,
                                                            auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        model.set_weights_index(0.9*w+0.1*w_source, -1)
        model.set_weights_index(0.9*v+0.1*v_source, -2)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w
def  conv_train_2_fc_layer_last_e2e_deep(model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)
    vfirst = model.get_weights_index(-3)
    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-2)
    # print('v',v.shape)
    # print(model.evaluate_both(train_loader, test_loader))
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -2)

        layer_in_first = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        # print(layer_in.shape, layer_in_first.shape)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()

        v, w, alpha_w_, lr = inc_solve_2_fc_layer_e2e_deep(model, i, False, layer_in_first, layer_in, layer_tar,
                                              vfirst,v, w, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        model.set_weights_index(w, -1)
        model.set_weights_index(v, -2)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w
def conv_train_deep_fc_layer_last_e2e(model, i,j,x,y, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)
    curr_layer_front = model.layers[-2]
    u = model.get_weights_index(-2)
    curr_layer_frontfront = model.layers[-3]
    v = model.get_weights_index(-3)
    # print(w.shape,u.shape,v.shape)
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
        gain_rate = gain_schedule(epoch, j)
            #             if j > 5 and auto:
            #                 auto = False
            #                 gain_ = alpha_w * gain_
        # for i, (x, y) in enumerate(train_loader):
        #     print('=========== batch', i + 1, '/', len(train_loader), '==========')
        layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -3)
        layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
        # print(layer_tar[0:14])
        # input()
        v, w, alpha_w_, lr = inc_solve_deep_fc_layer_e2e(model, i, False, layer_in, layer_tar,
                                              v, w,u, curr_layer_front.activations, curr_layer_after.activations,
                                              loop, ran_mix, 0.01,gain_rate=gain_rate, gain_=gain_, auto=auto)
        #             if j <= 5 and auto:
            #                 alpha_w = alpha_w + alpha_w_/6
        model.set_weights_index(w, -1)
        model.set_weights_index(v, -3)
            # print(model.evaluate_both(train_loader, test_loader))
            # acc_lst = model.evaluate_both(train_loader, test_loader)
            # print(acc_lst)
            # save model
            # if model_name != '_0':
            #     if acc_lst[1] > max_acc_test:
            #         max_acc_test = acc_lst[1]
            #         j_max_old = j_at_max
            #         j_at_max = j
            #     model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w

# update one chosen layer whetherits conv or fc
def inc_train_1_layer_error_based(model, at_layer, train_loader, test_loader, pool=True,last=False, epoch=20, loop=1,
                                  ran_mix=False):
    curr_layer = model.layers[at_layer]
    # print('curr_layer',curr_layer.name,curr_layer.weights)
    w = model.get_weights_index(at_layer)
    # print('curr_layer',w.shape,w)
    # print(model.evaluate_both(train_loader, test_loader))

    with torch.no_grad():
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            for i, (x, y) in enumerate(train_loader):

                if (i + 1) % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                #                 print(model.no_layers - at_layer - 1)
                #                 input()
                if last ==False:
                    layer_tar = model.backward_to_layer(layer_tar, model.no_layers - at_layer - 1)
                if curr_layer.name == 'conv':
                    pad = curr_layer.padding
                    conv_layer = curr_layer
                    if pool:
                        fc_w = model.get_weights_index(at_layer + 3)
                        pool_layer = model.layers[at_layer + 1]
                        flat_layer = model.layers[at_layer + 2]
                        w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                         flat_layer,
                                                         w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                    else:
                        fc_w = model.get_weights_index(at_layer + 2)
                        pool_layer = False
                        flat_layer = model.layers[at_layer + 1]
                        w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                         flat_layer,
                                                         w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                elif curr_layer.name == 'fc':
                    # print()
                    w = inc_solve_fc(i, False, layer_in, layer_tar,
                                     w, curr_layer.activations, loop, ran_mix, -1, gain_rate)
                    print('w',w)
                model.set_weights_index(w, at_layer)
                test0=model.get_weights_index(-2)
                test1=model.get_weights_index(-1)
                print(0,test0.size,test0)
                print(1, test1.size,test1)
            # print(model.evaluate_both(train_loader, test_loader))

    return w

def inc_train_1_layer_error_based_e2e(learnrate,model, at_layer, i,j,x,y, test_loader, pool=True,last=False, epoch=20, loop=1,
                                  ran_mix=False):
    curr_layer = model.layers[at_layer]
    # print('curr_layer',curr_layer.name,curr_layer.weights)
    w = model.get_weights_index(at_layer)
    # print('curr_layer',w.shape,w)
    # print(model.evaluate_both(train_loader, test_loader))

    with torch.no_grad():
        # for j in range(epoch):
        #     print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch,j)
            # print(gain_rate)
            # for i, (x, y) in enumerate(train_loader):

            # if (i + 1) % (len(train_loader) // 10) == 0:
            # print('=========== batch', i + 1, '/', '==========')
            layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
            layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
            #                 print(model.no_layers - at_layer - 1)
            #                 input()
            if last ==False:
                layer_tar = model.backward_to_layer(layer_tar, model.no_layers - at_layer - 1)
            if curr_layer.name == 'conv':
                pad = curr_layer.padding
                conv_layer = curr_layer
                if pool:
                    fc_w = model.get_weights_index(at_layer + 3)
                    pool_layer = model.layers[at_layer + 1]
                    flat_layer = model.layers[at_layer + 2]
                    w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                     flat_layer,
                                                     w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                else:
                    fc_w = model.get_weights_index(at_layer + 2)
                    pool_layer = False
                    flat_layer = model.layers[at_layer + 1]
                    w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                     flat_layer,
                                                     w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
            elif curr_layer.name == 'fc':
                # print()
                w ,error= inc_solve_fc(learnrate,i, False, layer_in, layer_tar,
                                 w, curr_layer.activations, loop, ran_mix, -1, gain_rate)
                err=0
                for i in range(len(error)):
                    # print(i)
                    err=err+error[i]*error[i]
                # print('error',err)
            model.set_weights_index(w, at_layer)
            # test0=model.get_weights_index(-2)
            # test1=model.get_weights_index(-1)
            # print(0,test0.size,test0)
            # print(1, test1.size,test1)
            # print(model.evaluate_both(train_loader, test_loader))

    return w,err

# UPDATE CONV LAYER WEIGHT
def inc_solve_filter_error_based(batch, lin, in_images, out_images, conv_layer, pool_layer, flat_layer,
                                 fil, fc_w, func, loop, ran_mix, gain_rate, pad):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)

        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)

        out_images = mf.fun_cut(out_images, func)

    in_shape = in_images.shape
    # out_shape = out_images.shape
    #
    # out_images = torch.reshape(out_images, [out_shape[0], out_shape[1], out_shape[2] * out_shape[3]])

    shape_filter = fil.shape
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights])

    matrix_x = create_matrix_x(in_images, fil, pad)

    gain = 1e-5
    lr = gain * gain_rate

    # print('init shapes ', matrix_x.shape, out_images.shape)

    if ran_mix:
        print('inc_solve_x_random_shuffle')
        matrix_x, out_images = data_randomize(matrix_x, out_images)

    for j in range(loop):

        if loop > 1:
            if j == math.ceil(loop / 2) + 1:
                lr = lr / 2
            if j == math.ceil(3 * loop / 4) + 1 & loop > 4:
                lr = lr / 2
            if j == loop - 1 & loop > 5:
                lr = lr / 5
            if j == loop & loop > 8:
                lr = lr / 10
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(in_shape[0]):
            w_old = fil_w
            in_matrix = matrix_x[k, :, :]
            y_fil = w_old @ in_matrix
            # print(y_fil.shape)
            y_fil_ = y_fil.reshape(conv_layer.shape)
            # print(y_fil_.shape)
            if pool_layer:
                y_fil__ = f.avg_pool2d(y_fil_, 2, 2)
            else:
                y_fil__ = y_fil_
            # print(y_fil__.shape)
            y_fil___ = y_fil__.reshape(flat_layer.shape).unsqueeze(0)
            # print(y_fil___.shape, fc_w.shape)
            y_ = fc_w @ torch.t(y_fil___)

            if ~lin:
                # print('nonlinear coming here')
                y_ = func(y_)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # Backward error
            e_bw = torch.t(fc_w) @ e_
            # print('e_bw', e_bw.shape)
            if pool_layer:
                e_bw_ = e_bw.reshape(pool_layer.shape).unsqueeze(0)
                # print('e_bw_', e_bw_.shape)
                e_bw__ = pool_backward_error(e_bw_, 2)
            else:
                e_bw__ = e_bw
            # print('e_bw__', e_bw__.shape)
            e_bw___ = e_bw__.reshape(conv_layer.shape[0], -1)
            # print(fil_w.shape, e_bw___.shape, in_matrix.shape)
            fil_w = w_old + lr * e_bw___ @ torch.t(in_matrix)

    x = torch.reshape(fil_w, shape_filter)

    return x

# update weight in two direction (backward pass and forward pass) layer by layer 
def inverse_layerwise_training_error_based(model, train_loader, test_loader,
                                           config, pool=True, no_layers=1, epoch=1, loop=1, mix_data=False):
    for i in range(no_layers):
        cur_inx = model.no_layers - i - 1
        # print('cur_inx is:', cur_inx)
        cur_lay = model.layers[cur_inx]
        print('First time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer_error_based(model, cur_inx, train_loader, test_loader, pool, epoch, loop,
                                                       mix_data)
            model.set_weights_index(out_weight, cur_inx)
    for i in range(no_layers - 2):
        cur_inx = i + model.no_layers - no_layers + 1
        cur_lay = model.layers[cur_inx]
        print('Second time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer_error_based(model, cur_inx, train_loader, test_loader, pool, epoch, loop,
                                                       mix_data)
            model.set_weights_index(out_weight, cur_inx)
    return 0
