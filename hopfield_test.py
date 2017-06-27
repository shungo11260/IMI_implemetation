
# coding: utf-8

# In[ ]:

import numpy as np
import random
import matplotlib.pyplot as plt


# In[ ]:

random.randrange(-1, 2, 2)


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

td_num = 1
for i in range(td_num):
    W = update_weight(W, train_data[i], i)


# In[ ]:



