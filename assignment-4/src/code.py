#!/usr/bin/env python
# coding: utf-8

# In[270]:


import os # processing file path
import gzip # unzip the .gz file, not used here
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[271]:


def split_dataset(data,shuf):
    X = data.loc[0:, data.columns != 'label']
    Y = data.loc[0:, 'label']
    X_train, X_test, Y_train, Y_test = train_test_split(  X, Y, test_size = 0,shuffle=shuf)
    return X, Y, X_train, X_test, Y_train, Y_test


# In[272]:


df = pd.read_csv('Apparel/apparel-trainval.csv',sep= ',')
X, Y, X_train, X_test, y_train, y_test = split_dataset(df,True)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
mask = list(range(59000, 59000 + 1000))
X_val = X_train[mask]
X_val = X_val.reshape(1000, -1)
y_val = y_train[mask]
mask = list(range(59000))
X_train = X_train[mask]
X_train = X_train.reshape(59000, -1)
y_train = y_train[mask]


# In[310]:


def ReLU(x):    
    return np.maximum(0, x)


# In[313]:


def sigmoid(x):    
    return 1 / (1 + math.exp(-x))


# In[342]:


class NeuralNet(object):    
    def __init__(self, input_size, hidden_size, output_size, std=1e-4): 
        self.params = {}
        sz1 = np.random.randn(input_size, hidden_size)
        sz2 = np.random.randn(hidden_size, output_size)
        sz3 = np.zeros((1, hidden_size))  
        sz4 = np.zeros((1, output_size))
        self.params['W1'] = std * sz1
        self.params['W2'] = std * sz2   
        self.params['b1'] = sz3    
        self.params['b2'] = sz4

    def loss(self, X, y=None, reg=0.0):
        N, D = X.shape
        scores = None
        W2 = self.params['W2']
        b2 = self.params['b2']
        W1 = self.params['W1']
        b1 = self.params['b1']
        ddt = np.dot(X, W1) + b1
        h1 = ReLU(ddt)
        ddt = np.dot(h1, W2)
        scores = ddt + b2 
        
        if y is None:   
            return scores
        
        scores_max = np.max(scores, axis=1, keepdims=True)    
        exp_scores = np.exp(scores - scores_max)
        
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)    
        correct_logprobs = -np.log(probs[range(N), y])        
        data_loss = np.sum(correct_logprobs)
        data_loss = data_loss / N
        ww1 = np.sum(W1*W1)
        ww2 = np.sum(W2*W2)
        loss = data_loss + 0.5 * reg * ww1 + 0.5 * reg * ww2
        
        grads = {}
        dscores = probs                                 
        dscores[range(N), y] -= 1
        dscores /= N
        db2 = np.sum(dscores, axis=0, keepdims=True)    
        dh1 = np.dot(dscores, W2.T)                     
        dh1[h1 <= 0] = 0
        
        grads['W1'] = np.dot(X.T, dh1) + reg * W1
        grads['W2'] = np.dot(h1.T, dscores) + reg * W2 
        grads['b1'] = np.sum(dh1, axis=0, keepdims=True)
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, 
               learning_rate_decay=0.95, reg=1e-5, num_epochs=10, 
               batch_size=200, verbose = False):   
        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train / batch_size), 1)
        
        v_W1, v_b1 = 0.0, 0.0
        v_W2, v_b2 = 0.0, 0.0
        lsh = []
        train_acc_history = []
        val_acc_history = []
        
        rng = num_epochs * iterations_per_epoch + 1
        
        for it in range(1, rng):   
            y_batch = None 
            X_batch = None   

            sample_index = np.random.choice(num_train, batch_size, replace=True)   
            X_batch = X[sample_index, :]          
            y_batch = y[sample_index]             
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg) 
            lsh.append(loss)
            tt1 = learning_rate * grads['W2'] 
            v_W2 = 0.9 * v_W2 - tt1    
            self.params['W2'] += 0.9 * v_W2 - tt1
            tt2 = learning_rate * grads['b2']
            self.params['b2'] += 0.9 * v_b2 - tt2 
            tt3 = learning_rate * grads['b1']
            self.params['b1'] += 0.9 * v_b1 - tt3
            tt4 = learning_rate * grads['W1']
            v_W1 = 0.9 * v_W1 - tt4   
            self.params['W1'] += v_W1   

            if verbose == True and it % iterations_per_epoch == 0:
                tmp = iterations_per_epoch
                train_acc = (self.predict(X_batch) == y_batch).mean()    
                val_acc = (self.predict(X_val) == y_val).mean()    
                train_acc_history.append(train_acc)    
                val_acc_history.append(val_acc) 
                epoch = it / tmp    
                learning_rate *= learning_rate_decay

        return {
            'train_acc_history': train_acc_history,   
            'lsh': lsh,   
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):    
        y_pred = None
        ddt = np.dot(X, self.params['W1']) + self.params['b1']
        h1 = ReLU(ddt)
        ddt1 = np.dot(h1, self.params['W2'])
        scrs =  ddt1 + self.params['b2']
        y_pred = np.argmax(scrs, axis=1)    
        return y_pred


# In[413]:


net = NeuralNet(X_train.shape[1], 10, 10)
stats = net.train(X_train, y_train, X_val, y_val,
            num_epochs=10, batch_size=1024,
            learning_rate=7.5e-4, learning_rate_decay=0.95,
            reg=1.0, verbose = True)

predict_val = net.predict(X_val)
cnt = 0
for i in range(len(predict_val)):
    if predict_val[i]==y_val[i]:
        cnt = cnt + 1


# In[414]:


givendataset = pd.read_csv('Apparel/apparel-test.csv')
testdataset = pd.read_csv('input/fashion-mnist_test.csv')
testdataset.drop(testdataset.tail(326).index,inplace=True)
X, Y, X_test, tmp1, y_test, tmp2 = split_dataset(testdataset,False)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)


# In[390]:


# hidden_size = 10
# num_classes = 10
# results = {}
# best_val = -1
# best_net = None

# learning_rates = np.array([2.5,5,7.5,10])*1e-4
# regularization_strengths = [0.25,0.5,0.75,1]

# for lr in learning_rates:
#     for reg in regularization_strengths:
#         net = NeuralNet(input_size, hidden_size, num_classes)
#         stats = net.train(X_train, y_train, X_val, y_val,
#         num_epochs=10, batch_size=1024,
#         learning_rate=lr, learning_rate_decay=0.95,
#         reg= reg, verbose=False)
#         val_acc = (net.predict(X_val) == y_val).mean()
#         if val_acc > best_val:
#             best_val = val_acc
#             best_net = net         
#         results[(lr,reg)] = val_acc


# for lr, reg in sorted(results):
#     val_acc = results[(lr, reg)]
#     print('lr %e reg %e val accuracy: %f' % (
#                 lr, reg,  val_acc))
    
# print('best validation accuracy achieved during cross-validation: %f' % best_val)


# In[403]:


arr = []
pred = best_net.predict(X_test)
for val in pred:
    arr.append([val])
np.savetxt("20161005_apparel_prediction5.csv", arr, delimiter=",", fmt='%s')


# In[404]:


tmp2 = []
for val in y_test:
    tmp2.append([val])
np.savetxt("tmp2.csv", tmp2, delimiter=",", fmt='%s')


# In[406]:


ppf1 = pd.read_csv('20161005_apparel_prediction5.csv')
ppf1 = np.asarray(ppf1)

ppf2 = pd.read_csv('tmp2.csv')
ppf2 = np.asarray(ppf2)


# In[407]:


cnt = 0
for i in range(len(ppf1)):
    if(ppf1[i]!=ppf2[i]):
        cnt = cnt + 1
cnt


# In[396]:


X_test2 = np.asarray(givendataset)
arr = []
pred = best_net.predict(X_test2)
for val in pred:
    arr.append([val])
np.savetxt("20161005_apparel_prediction6.csv", arr, delimiter=",", fmt='%s')


# In[ ]:




