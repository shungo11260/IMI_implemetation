import numpy as np
import matplotlib.pyplot as plt
import random
import hopfield as hf
from train_data import TrainData as td
import copy

td_num = 1
theta = np.zeros((25,1))
noise_percentage = 0.52

if __name__=="__main__":

    t = td()

    W = hf.init_weight(t.train_data)
    for i in range(td_num):
        W = hf.update_weight(W, t.train_data[i], i)

    input1_0 = copy.deepcopy(t.train_data1)
    input1_1 = copy.deepcopy(t.train_data1)
    input1_1 = hf.add_noise2td(input1_1, noise_percentage)

    real_noise_per = 0.0
    comp_accuracy = 0.0

    # calculate accurate noise percentage
    for i in range(input1_0.shape[0]):
        for j in range(input1_0.shape[0]):
            if  input1_0[i][j] != input1_1[i][j]:
                real_noise_per += 1.0
    real_noise_per /= 25.0

    hf.update(train_data=t.train_data1, input_data=input1_1, weight=W, threshold=theta, update_num=1000, limit_count=50)

    # calculate right component percentage
    for i in range(input1_0.shape[0]):
        for j in range(input1_0.shape[0]):
            if  input1_0[i][j] == input1_1[i][j]:
                comp_accuracy += 1.0
    comp_accuracy /= 25.0

    print(" real noise percentage:", real_noise_per)
    print("compare right compo percentage:", comp_accuracy)
