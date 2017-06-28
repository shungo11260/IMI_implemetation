
# coding: utf-8

# In[ ]:

import numpy as np
import random
import matplotlib.pyplot as plt


# In[ ]:

plt.subplot(1,6,1)
train_data1 = np.ones((5,5))
train_data1[0:5, ::4] = -1
train_data1[1, 1::2] = -1
train_data1[2, ::2] = -1
plt.imshow(train_data1, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,2)
train_data2 = np.ones((5,5))
train_data2[0, 2] = -1
train_data2[1, 1::2] = -1
train_data2[2:5, ::4] = -1
train_data2[3] = -1
plt.imshow(train_data2, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,3)
train_data3 = np.ones((5,5))
train_data3[0] = -1
train_data3[1:5, 2] = -1
plt.imshow(train_data3, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,4)
train_data4 = np.ones((5,5))
train_data4[0:5, ::3] = -1
train_data4[2, :4] = -1
plt.imshow(train_data4, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,5)
train_data5 = np.ones((5,5))
train_data5[::2] = -1
train_data5[0, 4] = 1
train_data5[1, 0] = -1
train_data5[3, 4] = -1
plt.imshow(train_data5, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,6)
train_data6 = np.ones((5,5))
train_data6[0:3, 1] = -1
train_data6[4, 1] = -1
plt.imshow(train_data6, interpolation='nearest', cmap='bone')


# In[ ]:

train_data = []
train_data.append(train_data1)
train_data.append(train_data2)
train_data.append(train_data3)
train_data.append(train_data4)
train_data.append(train_data5)
train_data.append(train_data6)


# In[ ]:

plt.show()


# In[ ]:

def set_weight(data):
    vector_size = data.size
    vector = data.reshape((vector_size, 1))
    weight = np.dot(vector, vector.T) - np.eye(25)
    return weight


# In[ ]:

def update_weight(weight, data, n):
    weight = (n*weight + set_weight(data)) / (n+1)
    return weight


# In[ ]:

# make weight as W
td_num = 1
W = np.zeros((25, 25))
for i in range(td_num):
    W = update_weight(W, train_data[i], i)


# In[ ]:

# make thresholod
theta = np.zeros((25,1))


# In[ ]:

def add_noise2td(train_data, percentage):
    per = round(percentage*train_data.size)
    data = train_data.reshape((train_data.size, 1))
    for k in range(per):
        n = random.randrange(0, 25)
        data[n] = random.randrange(-1, 2, 2)
    data = data.reshape((train_data.shape[0], train_data.shape[1]))
    return data


# In[ ]:

# make input data as tdx
td1 = add_noise2td(train_data1, 0.5)


# In[ ]:

plt.imshow(td1, interpolation='nearest', cmap='bone')
plt.show()


# In[ ]:

def lyapunov_function(input_data, weight, threshold):
    in_vec = input_data.reshape((25,1))
    V = -(np.dot(np.dot(in_vec.T, weight), in_vec)) + np.dot(threshold.T, in_vec)
    return V


# In[ ]:

def update_input_one_step(input_data, weight, threshold):
    x = input_data.reshape((25, 1))
    n = random.randrange(0, 25)
    x[n] = np.sign(np.dot(weight[n], x) - threshold[n])

    return x.reshape((5,5))

def update(input_data, weight, threshold, update_num):
    V_old = [[0.0]]
    for i in range(update_num):
        in_da = update_input_one_step(input_data, weight, threshold)
        if lyapunov_function(in_da, weight, threshold) == V_old:
            break
        else:
            plt.figure(i)
            V_old = lyapunov_function(in_da, weight, threshold)
            print(V_old)
            plt.imshow(in_da, interpolation='nearest', cmap='bone')
    plt.show()


# In[ ]:

update(td1, W, theta, 100)


# In[ ]:

# update input
td1 = update_input_one_step(td1, W)
plt.imshow(td1, interpolation='nearest', cmap='bone')
plt.show()


# In[ ]:

for i in range(10):
    td1 = update_input_one_step(td1, W)


# In[ ]:

x = x.reshape((5,5))


# In[ ]:

plt.imshow(x, interpolation='nearest', cmap='bone')
plt.show()


# In[ ]:

plt.imshow(td1, interpolation='nearest', cmap='bone')
plt.show()


# In[ ]:




# In[ ]:

x = np.zeros((5, 5))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i][j] = random.randrange(-1, 2, 2)
plt.imshow(x, interpolation='nearest', cmap='bone')
plt.show()


# In[ ]:



