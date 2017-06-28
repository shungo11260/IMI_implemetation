
# coding: utf-8

# In[ ]:

import numpy as np
import hopfield as hf
from train import TrainData
import matplotlib.pyplot as plt
td_num = 1


# In[ ]:

td = TrainData()


# In[ ]:

plt.subplot(1,6,1)
plt.imshow(td.train_data1, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,2)
plt.imshow(td.train_data2, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,3)
plt.imshow(td.train_data3, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,4)
plt.imshow(td.train_data4, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,5)
plt.imshow(td.train_data5, interpolation='nearest', cmap='bone')


# In[ ]:

plt.subplot(1,6,6)
plt.imshow(td.train_data6, interpolation='nearest', cmap='bone')


# In[ ]:

plt.show()


# In[ ]:

W = hf.init_weight(td.train_data)
for i in range(td_num):
    W = hf.update_weight(W, td.train_data[i], i)


# In[ ]:



