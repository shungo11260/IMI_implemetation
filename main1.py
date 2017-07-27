import numpy as np
import matplotlib.pyplot as plt
import random
import hopfield as hf
from train_data import TrainData as td

td_num = 2
theta = np.zeros((25,1))
noise_percentage = 0.2

if __name__=="__main__":

    t = td()

    W = hf.init_weight(t.train_data)
    for i in range(td_num):
        W = hf.update_weight(W, t.train_data[i], i)

    input1 = hf.add_noise2td(t.train_data2, noise_percentage)

    hf.update(train_data=t.train_data1, input_data=input1, weight=W, threshold=theta, update_num=100, limit_count=25)
