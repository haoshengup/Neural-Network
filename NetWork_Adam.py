#!/usr/bin/python
#description : A simple Python script of neural network
#author : haosheng
#date : 20180201
#version : 0.1
#usage : NetWork_Adam.py
#python_version : 3.6.3

import numpy as np
import math
from sklearn import datasets
from sklearn import preprocessing
import random

class NetWork(object):
    def __init__(self,sizes):
        '''
        initialize an object
        :param sizes: a list, the elemnt of the list is the num of nodes of every node
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(next_layer_num, previous_layer_num)/previous_layer_num \
                        for previous_layer_num, next_layer_num in zip(sizes[:-1], sizes[1:])]
        self.sum_lambda_weights = np.zeros_like(self.weights)
        self.weights_first_moment = np.zeros_like(self.weights)
        self.weights_second_moment = np.zeros_like(self.weights)

        self.gamma = [np.ones((node, 1)) for node in self.sizes]
        self.sum_lambda_gamma = np.zeros_like(self.gamma)
        self.gamma_first_moment = np.zeros_like(self.gamma)
        self.gamma_second_moment = np.zeros_like(self.gamma)

        self.beta = [np.zeros((node, 1)) for node in self.sizes]
        self.sum_lambda_beta = np.zeros_like(self.beta)
        self.beta_first_moment = np.zeros_like(self.beta)
        self.beta_second_moment = np.zeros_like(self.beta)

        self.epsilon = 0.001

    def forward(self, a, mode = 'train', e_mv_out = None):
        
        out = [a]
        net_list = [a]
        mv_list = [np.zeros(shape=(a.shape[0], 2))]  
        x_norm_list = [a]  
        if mode == 'train':
            for w,gamma, beta in zip(self.weights[:-1], self.gamma[1:-1], self.beta[1:-1]):
                net = np.dot(w, a)
                net_list.append(net)
                net_mean = net.mean(axis = 1)[:,np.newaxis]
                net_std = net.std(axis = 1)[:,np.newaxis]
                net_var = pow(net_std, 2)
                x_norm = (net - net_mean)/np.sqrt(net_var + self.epsilon)
                x_norm_list.append(x_norm)
                mv_list.append(np.hstack((net_mean, net_var)))
                y = gamma * x_norm + beta
                a = self.relu(y)
                out.append(a)
            net = np.dot(self.weights[-1], out[-1])
            net_list.append(net)
            net_mean = net.mean(axis=1)[:, np.newaxis]
            net_std = net.std(axis=1)[:, np.newaxis]
            net_var = pow(net_std, 2)
            x_norm = (net - net_mean) / np.sqrt(net_var + self.epsilon)
            x_norm_list.append(x_norm)
            mv_list.append(np.hstack((net_mean, net_var)))
            y = self.gamma[-1] * x_norm + self.beta[-1]
            out.append(self.softmax(y))
            return  out, net_list, x_norm_list, mv_list
        else:
            for w, gamma, beta, e_mv in zip(self.weights[:-1], self.gamma[1:-1], self.beta[1:-1], e_mv_out[1:-1]):
                net = np.dot(w, a)
                x_norm = (net - e_mv[:,0][:,np.newaxis])/np.sqrt(e_mv[:,1][:,np.newaxis] + self.epsilon)
                y = gamma * x_norm + beta
                a = self.relu(y)
                out.append(a)
            net = np.dot(self.weights[-1], out[-1])
            x_norm = (net - e_mv_out[-1][:,0][:,np.newaxis]) / np.sqrt(e_mv_out[-1][:,1][:,np.newaxis] + self.epsilon)
            y = self.gamma[-1] * x_norm + self.beta[-1]
            out.append(self.softmax(y))
            return out

    def backward(self, out_forward, target):
        
        delta = [np.zeros((node, target.shape[1])) for node in self.sizes]
        y_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]
        x_norm_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]
        var_derivative = [np.zeros((node, 1)) for node in self.sizes]
        mean_derivative = [np.zeros((node, 1)) for node in self.sizes]
        net_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]

        y_derivative[-1] = (self.cost_derivative(out_forward[0][-1], target)) * self.softmax_derivative(out_forward[0][-1])
        x_norm_derivative[-1] = y_derivative[-1] * self.gamma[-1]
        var_derivative[-1] = (x_norm_derivative[-1] * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis]) * (-0.5) * pow(out_forward[3][-1][:, 1][:,np.newaxis] + self.epsilon, -1.5)).sum(axis=1)[:, np.newaxis]
        mean_derivative[-1] = (x_norm_derivative[-1] * (-1) / np.sqrt(out_forward[3][-1][:, 1][:, np.newaxis] + self.epsilon)).sum(axis=1)[:, np.newaxis] \
                              + var_derivative[-1] * (-2 * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis])).sum(axis=1)[:, np.newaxis] / target.shape[1]
        net_derivative[-1] = x_norm_derivative[-1] / np.sqrt(out_forward[3][-1][:, 1][:, np.newaxis] + self.epsilon) + \
                             var_derivative[-1] * 2 * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis]) / target.shape[1] \
                              + mean_derivative[-1] / target.shape[1]

        
        delta[-1] = net_derivative[-1]
       
        for i in reversed(range(self.num_layers - 1)):
            y_derivative[i] = np.dot(self.weights[i].T, delta[i + 1]) * self.relu_derivative(out_forward[0][i])
            x_norm_derivative[i] = y_derivative[i] * self.gamma[i]
            var_derivative[i] = (x_norm_derivative[i] * (out_forward[1][i] - out_forward[3][i][:, 0][:,np.newaxis]) * (-0.5) * pow(out_forward[3][i][:,1][:,np.newaxis] + self.epsilon, -1.5)).sum(axis=1)[:,np.newaxis]
            mean_derivative[i] = (x_norm_derivative[i] * (-1) / np.sqrt(out_forward[3][i][:, 1][:,np.newaxis] + self.epsilon)).sum(axis=1)[:,np.newaxis] \
                                  + var_derivative[i] * (-2 * (out_forward[1][i] - out_forward[3][i][:,0][:,np.newaxis])).sum(axis=1)[:,np.newaxis]/target.shape[1]
            net_derivative[i] = x_norm_derivative[i] / np.sqrt(out_forward[3][i][:, 1][:,np.newaxis] + self.epsilon) + var_derivative[i] * 2 * (out_forward[1][i] - out_forward[3][i][:,0][:,np.newaxis])/target.shape[1] + mean_derivative[i]/target.shape[1]
            delta[i] = net_derivative[i]
        
		# calculate the derivative of weights
        lambda_weights = [np.dot(delta[i + 1], out_forward[0][i].T)/target.shape[1] for i in range(self.num_layers - 1)]
        self.sum_lambda_weights = [sw + lw for sw, lw in zip(self.sum_lambda_weights, lambda_weights)]

        # calculate the derivative of gamma
        lambda_gamma = [(yd * xm).sum(axis=1)[:, np.newaxis]/target.shape[1] for yd, xm in zip(y_derivative, out_forward[2])]
        self.sum_lambda_gamma = [sg + lg for sg, lg in zip(self.sum_lambda_gamma, lambda_gamma)]

        # calculate the derivative of beta
        lambda_beta = [yd.sum(axis=1)[:, np.newaxis]/target.shape[1] for yd in y_derivative]
        self.sum_lambda_beta = [sb + lb for sb, lb in zip(self.sum_lambda_beta, lambda_beta)]

    def Adam(self, training_data, target, eta, epoches, mini_batch_size):
        '''
        use Adam algorithm to train network
        :param training_data: training data, row: dimension of data, column: num of data
        :param target: target data，row: dimension of target, column: num of target
        :param eta: a list, the lenth of the list is 3. list[0]:learning rate，list[1]:momentum rate，list[2]:adagrad rate
        :param epoches: num of iteration
        :param mini_batch_size: the nums of trianing data in every mini_batch
        :return: modified weights and biases
        '''
        n = training_data.shape[1]
        index = [i for i in range(n)]
        random.shuffle(index)
        training_data = training_data[:,index]
        target = target[:,index]
        mini_batches = [(training_data[:,k:k + mini_batch_size], target[:,k:k + mini_batch_size]) for k in range(0,n,mini_batch_size)]
        e_mv_list = []
        for j in range(epoches):
            cost_total = 0
            if j < (epoches - 1):
                for mini_batch in mini_batches:
                    out_forward = self.forward(mini_batch[0])
                    self.backward(out_forward, mini_batch[1])
                    cost_total = cost_total + self.cost(out_forward[0][-1], mini_batch[1])

                self.weights_first_moment = [eta[1] * wfm + (1 - eta[1]) * slw for wfm, slw in zip(self.weights_first_moment, self.sum_lambda_weights)]
                self.gamma_first_moment = [eta[1] * gfm + (1 - eta[1]) * slg for gfm, slg in zip(self.gamma_first_moment, self.sum_lambda_gamma)]
                self.beta_first_moment = [eta[1] * bfm + (1 - eta[1]) * slb for bfm, slb in zip(self.beta_first_moment, self.sum_lambda_beta)]

                weights_first_unbias = [wfm / (1 - eta[1] ** (j + 1)) for wfm in self.weights_first_moment]
                gamma_first_unbias = [gfm / (1 - eta[1] ** (j + 1)) for gfm in self.gamma_first_moment]
                beta_first_unbias = [bfm / (1 - eta[1] ** (j + 1)) for bfm in self.beta_first_moment]

                self.weights_second_moment = [eta[2] * wsm + (1 - eta[2]) * slw * slw for wsm, slw in zip(self.weights_second_moment, self.sum_lambda_weights)]
                self.gamma_second_moment = [eta[2] * gsm + (1 - eta[2]) * slg * slg for gsm, slg in zip(self.gamma_second_moment, self.sum_lambda_gamma)]
                self.beta_second_moment = [eta[2] * bsm + (1 - eta[2]) * slb * slb for bsm, slb in zip(self.beta_second_moment, self.sum_lambda_beta)]

                weights_second_unbias = [wsm / (1 - eta[2] ** (j + 1)) for wsm in self.weights_second_moment]
                gamma_second_unbias = [gsm / (1 - eta[2] ** (j + 1)) for gsm in self.gamma_second_moment]
                beta_second_unbias = [bsm / (1 - eta[2] ** (j + 1)) for bsm in self.beta_second_moment]


                self.weights = [w - eta[0] * wfu / (np.sqrt(wsu) + 0.0000001) for w, wfu, wsu in zip(self.weights, weights_first_unbias, weights_second_unbias)]
                self.gamma = [g - eta[0] * gfu / (np.sqrt(gsu) + 0.0000001) for g, gfu, gsu in zip(self.gamma, gamma_first_unbias, gamma_second_unbias)]
                self.beta = [be - eta[0] * bfu / (np.sqrt(bsu) + 0.0000001) for be, bfu, bsu in zip(self.beta, beta_first_unbias, beta_second_unbias)]

            else:
                for mini_batch in mini_batches:
                    out_forward = self.forward(mini_batch[0])
                    self.backward(out_forward, mini_batch[1])
                    cost_total = cost_total + self.cost(out_forward[0][-1], mini_batch[1])
                    e_mv_list.append(np.array(out_forward[3]))

                self.weights_first_moment = [eta[1] * wfm + (1 - eta[1]) * slw for wfm, slw in zip(self.weights_first_moment, self.sum_lambda_weights)]
                self.gamma_first_moment = [eta[1] * gfm + (1 - eta[1]) * slg for gfm, slg in zip(self.gamma_first_moment, self.sum_lambda_gamma)]
                self.beta_first_moment = [eta[1] * bfm + (1 - eta[1]) * slb for bfm, slb in zip(self.beta_first_moment, self.sum_lambda_beta)]

                weights_first_unbias = [wfm / (1 - eta[1] ** j) for wfm in self.weights_first_moment]
                gamma_first_unbias = [gfm / (1 - eta[1] ** j) for gfm in self.gamma_first_moment]
                beta_first_unbias = [bfm / (1 - eta[1] ** j) for bfm in self.beta_first_moment]

                self.weights_second_moment = [eta[2] * wsm + (1 - eta[2]) * slw * slw for wsm, slw in zip(self.weights_second_moment, self.sum_lambda_weights)]
                self.gamma_second_moment = [eta[2] * gsm + (1 - eta[2]) * slg * slg for gsm, slg in zip(self.gamma_second_moment, self.sum_lambda_gamma)]
                self.beta_second_moment = [eta[2] * bsm + (1 - eta[2]) * slb * slb for bsm, slb in zip(self.beta_second_moment, self.sum_lambda_beta)]

                weights_second_unbias = [wsm / (1 - eta[2] ** j) for wsm in self.weights_second_moment]
                gamma_second_unbias = [gsm / (1 - eta[2] ** j) for gsm in self.gamma_second_moment]
                beta_second_unbias = [bsm / (1 - eta[2] ** j) for bsm in self.beta_second_moment]

                self.weights = [w - eta[0] * wfu / (np.sqrt(wsu) + 0.0000001) for w, wfu, wsu in zip(self.weights, weights_first_unbias, weights_second_unbias)]
                self.gamma = [g - eta[0] * gfu / (np.sqrt(gsu) + 0.0000001) for g, gfu, gsu in zip(self.gamma, gamma_first_unbias, gamma_second_unbias)]
                self.beta = [be - eta[0] * bfu / (np.sqrt(bsu) + 0.0000001) for be, bfu, bsu in zip(self.beta, beta_first_unbias, beta_second_unbias)]


            print('Epoch {0}: cost = {1}'.format(j, cost_total/n))
        e_mv_array = np.array(e_mv_list)
        e_mv_out = e_mv_array.mean(axis=0)
        e_mv_out = e_mv_out.tolist()
        for k in range(len(e_mv_out)):
            e_mv_out[k][:,1] = e_mv_out[k][:,1] * mini_batch_size / (mini_batch_size - 1)
        return e_mv_out

    def evaluate(self, test_data, test_target, e_mv_out):
        '''
        evaluate the test_data
        :param test_data: test data
        :param test_target: target
        :return: nums to identified correctly, nums to identified wrongly, recogniton rate
        '''
        out = self.forward(test_data, mode='test',e_mv_out=e_mv_out)
        out_max_index = np.argmax(out[-1], axis=0)
        total_num = test_target.shape[1]
        correct_num = np.sum((out_max_index == np.argmax(test_target, axis=0)) == 1)
        print('correct num：{0}\n'.format(correct_num))
        print('wrong num：{0}\n' .format(total_num - correct_num))
        print('recogniton rate：{0}\n' .format(correct_num/total_num))

    def relu(self, x):
        return np.where(x>0, x, 0)

    def relu_derivative(self, x):
        return np.where(x>0, 1, 0)

    def cost_derivative(self, out, target):
        return (out - target)

    def cost(self, out, target):
        return  sum(0.5 * sum((out - target) ** 2))

    def softmax(self, net):
        return np.exp(net)/sum(np.exp(net))

    def softmax_derivative(self, out):
        return out * (1 - out)

    def normalization(self,input_data,scaler = None):
        '''
        normalization as row
        :param input_data: input matrix
        :param scaler: parameter of normalization，scaler.mean_: mean of row，scaler.std_:standard deviation
        :return: a turple（normalization_out,scaler)，normalization_out:outcome，scaler:parameter of normalization
        '''
        input_data = input_data.T
        if not scaler:
            scaler = preprocessing.StandardScaler().fit(input_data)
        normalization_out = scaler.transform(input_data).T
        return normalization_out, scaler