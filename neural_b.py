import numpy as np
import pandas as pd
import math
import sys

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

def forward(X,w,af,lf):
    z = []
    z0 = X
    z0 = np.c_[np.ones(X.shape[0]),z0]
    z.append(z0)
    l = len(w)
    i = 0
    while(i < l-1):
        z0 = np.matmul(z0,w[i])
        if(af == 0):
            z0 = sigmoid(z0)
        elif(af == 1):
            z0 = tanh(z0)
        elif(af == 2):
            z0 = relu(z0)
        z0 = np.c_[np.ones(X.shape[0]),z0]
        z.append(z0)
        i = i+1
    z0 = np.matmul(z0,w[l-1])
    if(lf == 0):
        z0 = softmax(z0)
    if(lf == 1):
        if(af == 0):
            z0 = sigmoid(z0)
        elif(af == 1):
            z0 = tanh(z0)
        elif(af == 2):
            z0 = relu(z0)
    z.append(z0)
    return z

def backward(z,w,ytrain,n0,si,af,lf):
    l = len(z)
    p = z[l-1]-ytrain
    if(lf == 1):
        b = np.multiply(z[l-1],1-z[l-1])
        if(af == 1):
            b = 1-np.square(z[l-1])
        if(af == 2):
            b = np.where(z[l-1] == 0,0,1)
        p = np.multiply(p, b)
    r = z[l-2].T
    g = np.matmul(r,p)/si
    q = np.multiply(z[l-2],1-z[l-2])
    if(af == 1):
        q = 1-np.square(z[l-2])
    elif(af == 2):
        q = np.where(z[l-2] == 0,0,1)
    ans = []
    wg = np.asarray(w[l-2])
    wt = wg.T
    p = np.matmul(p,wt)
    p = np.multiply(p,q)
    p = p[:,1:]
    wg = wg-n0*g
    ans.append(wg)
    i = l-2
    while(i > 0):
        r = z[i-1].T
        g = np.matmul(r,p)/si
        q = np.multiply(z[i-1],1-z[i-1])
        if(af == 1):
            q = 1-np.square(z[i-1])
        elif(af == 2):
            q = np.where(z[i-1] == 0,0,1)
        wg = np.asarray(w[i-1])
        wt = wg.T
        p = np.matmul(p,wt)
        p = np.multiply(p,q)
        p = p[:,1:]
        wg = wg-n0*g
        ans.append(wg)
        i = i-1
    ans.reverse()
    return ans

input_path = sys.argv[1]
output_path = sys.argv[2]
param = sys.argv[3]
s = input_path+"train_data_shuffled.csv"
dftrain = pd.read_csv(s,header = None)
s = input_path+"public_test.csv"
dftest = pd.read_csv(s,header = None)
ty = pd.read_csv(param,header = None,delimiter = "\t")

epochs = int(ty[0][0])
batch_size = int(ty[0][1])
layer = ty[0][2]
lty = int(ty[0][3])
n0 = float(ty[0][4])
af = int(ty[0][5])
lf = int(ty[0][6])
sv = int(ty[0][7])
layer = layer[1:-1]
layer = layer.split(',')
np.random.seed(sv)

ytrain = dftrain.iloc[:,-1]
Xtrain = np.asarray(dftrain.iloc[:,:-1])
Xtest = np.asarray(dftest.iloc[:,:-1])
ytrain = np.asarray(pd.get_dummies(ytrain))
Xtrain = Xtrain/255
Xtest = Xtest/255

w = []

m = Xtrain.shape[1]
l = len(layer)
i = 0
while(i < l):
    x = int(layer[i])
    h = math.sqrt(2/(m+1+x))
    w1 = np.float32(np.random.normal(size=(m+1,x))*h)
    w.append(w1)
    m = x
    i = i+1

n = Xtrain.shape[0]
i = 1
while(i <= epochs):
    lrnig = n0
    if(lty == 1):
        lrnig = n0/math.sqrt(i)
    k = batch_size
    j = 0
    while((j+1)*k <= n):
        X = Xtrain[j*k:(j+1)*k]
        y = ytrain[j*k:(j+1)*k]
        z = forward(X,w,af,lf)
        w = backward(z,w,y,lrnig,k,af,lf)
        j = j+1
    i = i+1

l = len(w)
i = 0
while(i < l):
    s = output_path+"w_"+str(i+1)+".npy"
    q = np.asarray(w[i])
    np.save(s,q)
    i = i+1

z = forward(Xtest,w,af,lf)
l = len(z)
d = np.asarray(z[l-1])
ans = np.argmax(d,axis=1)
s = output_path+"predictions.npy"
np.save(s,ans)
