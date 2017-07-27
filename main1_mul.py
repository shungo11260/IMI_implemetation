import numpy as np
import matplotlib.pyplot as plt
import random
import hopfield as hf
from train_data import TrainData as td
import copy

td_num = 6
theta = np.zeros((25,1))
noise_percentage = 0.0
times = 1
update_num = 10000
limit_count = 25

if __name__=="__main__":

    t = td()

    W = hf.init_weight(t.train_data)
    for i in range(td_num):
        W = hf.update_weight(W, t.train_data[i], i)

    input1_0 = copy.deepcopy(t.train_data1)

    real_noise_per = np.array([0.0]*times)
    comp_accuracy = np.array([0.0]*times)

    # for result
    completely_right_number = 0.0   # seitou ritu
    nearly_average = 0.0    #ruiji do

    for k in range(times):
        input1_1 = copy.deepcopy(t.train_data1)
        input1_1 = hf.add_noise2td(input1_1, noise_percentage)

        # calculate accurate noise percentage
        for i in range(input1_0.shape[0]):
            for j in range(input1_0.shape[0]):
                if  input1_0[i][j] != input1_1[i][j]:
                    real_noise_per[k] += 1.0
        real_noise_per[k] /= 25.0

        hf.update(input_data=input1_1, weight=W, threshold=theta, update_num=update_num, limit_count=limit_count)

        # calculate right component percentage
        for i in range(input1_0.shape[0]):
            for j in range(input1_0.shape[0]):
                if  input1_0[i][j] == input1_1[i][j]:
                    comp_accuracy[k] += 1.0
        comp_accuracy[k] /= 25.0

        #print(" real noise percentage:", real_noise_per[k])
        #print("compare right compo percentage:", comp_accuracy[k])

        if comp_accuracy[k] == 1.0:
            completely_right_number += 1.0
        nearly_average += comp_accuracy[k]

    print(" completely_right_number:", 100*completely_right_number/times)
    print("nearly_average:", 100*nearly_average/times)
