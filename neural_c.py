import numpy as np
import pandas as pd
import math
import sys
import time

def loss(X,Y):
    ga = math.pow(10,-15)
    r = X.shape[0]
    X = np.log(np.clip(X,ga,1-ga))
    Y = np.multiply(Y,X)
    ans = -np.sum(Y)/r
    return ans

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

curr = time.time()
input_path = sys.argv[1]
output_path = sys.argv[2]
param = sys.argv[3]
ty = pd.read_csv(param,header = None,delimiter = "\t")
s = input_path+"train_data_shuffled.csv"
dftrain = pd.read_csv(s,header = None)
ytrain = dftrain.iloc[:,-1]
Xtrain = np.asarray(dftrain.iloc[:,:-1])
Xtrain = Xtrain/255
ytrain = np.asarray(pd.get_dummies(ytrain))
layer = ty[0][0]
n = Xtrain.shape[0]
np.random.seed(1)
epochs = 13
batch_size = 500
lty = 0
lv = 0.001
af = 2
lis = 3
sdval = 1
answ = [13,500,0,0.001,2,3,1]
if(layer == "[256,46]"):
    epochs = 35
    batch_size = 500
    lty = 0
    lv = 0.001
    af = 2
    lis = 3
    sdval = 1
    answ = [35,500,0,0.001,2,3,1]
sre = output_path+"my_params.txt"
sol = open(sre,"w")
for obj in answ:
    sol.write(str(obj)+"\n")
sol.close()
w = []
m = Xtrain.shape[1]
layer = layer[1:-1]
layer = layer.split(',')
l = len(layer)
i = 0
while(i < l):
    x = int(layer[i])
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
curr1 = time.time()
while(i <= epochs and time.time()-curr < 270):
    k = batch_size
    j = 0
    while((j+1)*k <= n and time.time()-curr < 270):
        X = Xtrain[j*k:(j+1)*k]
        y = ytrain[j*k:(j+1)*k]
        z = forward(X,w)
        w = backward(z,w,y,k,vt,mt)
        curr2 = time.time()
        if(curr2-curr1 > 60 and curr2-curr < 270):
            l = len(w)
            pj = 0
            while(pj < l):
                s = output_path+"w_"+str(pj+1)+".npy"
                q = np.asarray(w[pj])
                np.save(s,q)
                pj = pj+1
            curr1 = curr2
        j = j+1
    i = i+1

curr2 = time.time()
if(curr2-curr < 285):
    l = len(w)
    i = 0
    while(i < l):
        s = output_path+"w_"+str(i+1)+".npy"
        q = np.asarray(w[i])
        np.save(s,q)
        i = i+1