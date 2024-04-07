import numpy as np

class MLKNN():
    def __init__(self, k=10, s = 1):
         self._k = k
         self.s = s
        
    def knn(self,train_x, t_index):
        data_num = train_x.shape[0]
        dis = np.zeros(data_num)
        neighbors = np.zeros(self._k)
        for i in range(data_num):
            dis[i] = ((train_x[i] - train_x[t_index]) ** 2).sum()
            
        for i in range(self._k):
            temp = float('inf')
            temp_j = 0
            for j in range(data_num):
                if (j != t_index) and (dis[j] < temp):
                    temp = dis[j]
                    temp_j = j
            dis[temp_j] = float('inf')
            neighbors[i] = temp_j

        return neighbors

    def knn_test(self, train_x, t):
        data_num = train_x.shape[0]
        dis = np.zeros(data_num)
        neighbors = np.zeros(self._k)

        for i in range(data_num):
            dis[i] = ((train_x[i] - t) ** 2).sum()
        
        for i in range(self._k):

            temp = float('inf')
            temp_j = 0
            for j in range(data_num):
                if dis[j] < temp:
                    temp = dis[j]
                    temp_j = j
            dis[temp_j] = float('inf')
            neighbors[i] = temp_j

        return neighbors

    def fit(self,train_x, train_y):

        label_num = train_y.shape[1]
        train_data_num = train_x.shape[0]
        Ph1 = np.zeros(label_num)
        Ph0 = np.zeros(label_num)
        Peh1 = np.zeros([label_num, self._k + 1])
        Peh0 = np.zeros([label_num, self._k + 1])
        
        for i in range(label_num):
            cnt = 0
            for j in range(train_data_num):
                if train_y[j][i] == 1:
                    cnt = cnt + 1
            Ph1[i] = (self.s + cnt) / (self.s * 2 + train_data_num)
            Ph0[i] = 1 - Ph1[i]

        for i in range(label_num):

            c1 = np.zeros(self._k + 1)
            c0 = np.zeros(self._k + 1)
            for j in range(train_data_num):
                temp = 0
                neighbors = self.knn(train_x, j)
                for k1 in range(self._k):
                    temp = temp + int(train_y[int(neighbors[k1])][i])

                if train_y[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1
        
            for j in range(self._k + 1):
                Peh1[i][j] = (self.s + c1[j]) / (self.s * (self._k + 1) + np.sum(c1))
                Peh0[i][j] = (self.s + c0[j]) / (self.s * (self._k + 1) + np.sum(c0))
                
        return Ph1, Ph0, Peh1, Peh0
    
    def predict(self, train_x, test_y, test_x, train_y, Ph1, Ph0, Peh1, Peh0):
        predict = np.zeros(test_y.shape, dtype=np.int64)
        test_data_num = test_x.shape[0]
        Outputs = np.zeros(shape=[test_y.shape[0], test_y.shape[1]])
        label_num = train_y.shape[1]

        for i in range(test_data_num):
            neighbors = self.knn_test(train_x, test_x[i])

            for j in range(label_num):
                temp = 0
                for nei in neighbors:
                    temp = temp + int(train_y[int(nei)][j])

                Prob_in=Ph1[j] * Peh1[j][temp]
                Prob_out=Ph0[j] * Peh0[j][temp]

                if(Prob_in+Prob_out==0):
                        Outputs[i][j]=Ph1[j]
                else:
                        Outputs[i][j]=Prob_in/(Prob_in+Prob_out)
                
                if(Prob_in > Prob_out):
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0
        
        return Outputs, predict
                