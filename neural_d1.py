import numpy as np
import pandas as pd
import math
import sys
import time

def sigmoid(X):
    X = 1/(1+np.exp(-X))
    return X

def softmax(X):
    Y = np.exp(X-np.max(X,axis=1).reshape(-1,1))
    d = np.sum(Y,axis=1).reshape(-1,1)
    X = Y/d
    return X

def tanh(X):
    ans = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    return ans

def relu(X):
    ans = np.where(X > 0,X,0)
    return ans

def forward(X,w):
    z = []
    z0 = X
    z0 = np.c_[np.ones(X.shape[0]),z0]
    z.append(z0)
    l = len(w)
    i = 0
    while(i < l-1):
        z0 = np.matmul(z0,w[i])
        z0 = relu(z0)
        z0 = np.c_[np.ones(X.shape[0]),z0]
        z.append(z0)
        i = i+1
    z0 = np.matmul(z0,w[l-1])
    z0 = softmax(z0)
    z.append(z0)
    return z

def backward(z,w,ytrain,si,vt,mt):
    n0 = 0.001
    l = len(z)
    p = z[l-1]-ytrain
    r = z[l-2].T
    g = np.matmul(r,p)/si
    q = np.where(z[l-2] == 0,0,1)
    ans = []
    wg = np.asarray(w[l-2])
    wt = wg.T
    p = np.matmul(p,wt)
    p = np.multiply(p,q)
    p = p[:,1:]
    sd = np.square(g)
    vt[l-2] = 0.9*vt[l-2]+0.1*sd
    ep = math.pow(10,-8)
    sd = np.sqrt(vt[l-2]+ep)
    num = n0/sd
    wg = wg-np.multiply(num,g)
    ans.append(wg)
    i = l-2
    while(i > 0):
        r = z[i-1].T
        g = np.matmul(r,p)/si
        q = np.where(z[i-1] == 0,0,1)
        wg = np.asarray(w[i-1])
        wt = wg.T
        p = np.matmul(p,wt)
        p = np.multiply(p,q)
        p = p[:,1:]
        sd = np.square(g)
        vt[i-1] = 0.9*vt[i-1]+0.1*sd
        ep = math.pow(10,-8)
        sd = np.sqrt(vt[i-1]+ep)
        num = n0/sd
        wg = wg-np.multiply(num,g)
        ans.append(wg)
        i = i-1
    ans.reverse()
    return ans

input_path = sys.argv[1]
output_path = sys.argv[2]
s = input_path+"train_data_shuffled.csv"
dftrain = pd.read_csv(s,header = None)
ytrain = dftrain.iloc[:,-1]
Xtrain = np.asarray(dftrain.iloc[:,:-1])
s = input_path+"public_test.csv"
dftest = pd.read_csv(s,header = None)
Xtest = np.asarray(dftest.iloc[:,:-1])
Ytest = np.asarray(dftest.iloc[:,-1])
Xtest = Xtest/255
Xtrain = Xtrain/255
ytrain = np.asarray(pd.get_dummies(ytrain))
layer = [[512,46],[256,46],[128,46],[64,46],[512,256,46],[256,64,46],[512,256,128,46],[512,128,64,46]]
n = Xtrain.shape[0]
np.random.seed(1)
epochs = 36
acc = []
jk = 0
while(jk < 8):
    w = []
    m = Xtrain.shape[1]
    l = len(layer[jk])
    i = 0
    while(i < l):
        x = int(layer[jk][i])
        h = math.sqrt(2/(m+1+x))
        w1 = np.float32(np.random.normal(size=(m+1,x))*h)
        w.append(w1)
        m = x
        i = i+1
    vt = []
    mt = []
    l = len(w)
    i = 0
    while(i < l):
        r,c = w[i].shape
        kl = np.zeros((r,c))
        vt.append(kl)
        mt.append(kl)
        i = i+1
    i = 1
    curr = time.time()
    while(i <= epochs and time.time()-curr < 270):
        k = 500
        j = 0
        while((j+1)*k <= n):
            X = Xtrain[j*k:(j+1)*k]
            y = ytrain[j*k:(j+1)*k]
            z = forward(X,w)
            w = backward(z,w,y,k,vt,mt)
            j = j+1
        i = i+1
    z = forward(Xtest,w)
    l = len(z)
    d = np.asarray(z[l-1])
    ans = np.argmax(d,axis=1)
    pred = np.sum(np.where(ans==Ytest,1,0))/Ytest.shape[0]
    acc.append(pred)
    jk = jk+1

stri = "Architecture"
dfk = pd.DataFrame(list(zip(layer,acc)),columns=['Architecture','Accuracy'])
op = output_path+stri+".csv"
dfk.to_csv(op,index=False)