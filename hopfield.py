import numpy as np
import random
import matplotlib.pyplot as plt

def init_weight(data):
        weight = np.ones((data[0].size, data[0].size))
        return weight

def set_weight(data):
    vector_size = data.size
    vector = data.reshape((vector_size, 1))
    weight = np.dot(vector, vector.T) - np.eye(25)
    return weight

def update_weight(weight, data, n):
    weight = (n*weight + set_weight(data)) / (n+1)
    return weight

def add_noise2td(train_data, percentage):
    per = round(percentage*train_data.size)
    data = train_data.reshape((train_data.size, 1))
    for k in range(per):
        n = random.randrange(0, 25)
        data[n] = random.randrange(-1, 2, 2)
    data = data.reshape((train_data.shape[0], train_data.shape[1]))
    return data

def lyapunov_function(input_data, weight, threshold):
    in_vec = input_data.reshape((25,1))
    V = -(np.dot(np.dot(in_vec.T, weight), in_vec)) + np.dot(threshold.T, in_vec)
    return V

def update_input_one_step(input_data, weight, threshold):
    x = input_data.reshape((25, 1))
    n = random.randrange(0, 25)
    x[n] = np.sign(np.dot(weight[n], x) - threshold[n])

    return x.reshape((5,5))

def update(train_data, input_data, weight, threshold, update_num, limit_count):
    V_old = [[0.0]]
    count = 0
    for i in range(update_num):
        in_da = update_input_one_step(input_data, weight, threshold)
        if lyapunov_function(train_data, weight, threshold) == V_old:
            print(i)
            print(lyapunov_function(train_data, weight, threshold))
            count += 1
            if count == limit_count:
                count = 0
                break
        else:
            print(i)
            count = 0
            plt.figure(i)
            V_old = lyapunov_function(in_da, weight, threshold)
            print(V_old)
            plt.imshow(in_da, interpolation='nearest', cmap='bone')
    plt.show()
