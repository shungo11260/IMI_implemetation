import numpy as np

class TrainData:
    def __init__(self):
        self.train_data1 = np.ones((5,5))
        self.train_data1[0:5, ::4] = -1
        self.train_data1[1, 1::2] = -1
        self.train_data1[2, ::2] = -1

        self.train_data2 = np.ones((5,5))
        self.train_data2[0, 2] = -1
        self.train_data2[1, 1::2] = -1
        self.train_data2[2:5, ::4] = -1
        self.train_data2[3] = -1

        self.train_data3 = np.ones((5,5))
        self.train_data3[0] = -1
        self.train_data3[1:5, 2] = -1

        self.train_data4 = np.ones((5,5))
        self.train_data4[0:5, ::3] = -1
        self.train_data4[2, :4] = -1

        self.train_data5 = np.ones((5,5))
        self.train_data5[::2] = -1
        self.train_data5[0, 4] = 1
        self.train_data5[1, 0] = -1
        self.train_data5[3, 4] = -1

        self.train_data6 = np.ones((5,5))
        self.train_data6[0:3, 1] = -1
        self.train_data6[4, 1] = -1

        self.train_data = []
        self.train_data.append( self.train_data1)
        self.train_data.append( self.train_data2)
        self.train_data.append( self.train_data3)
        self.train_data.append( self.train_data4)
        self.train_data.append( self.train_data5)
        self.train_data.append( self.train_data6)
