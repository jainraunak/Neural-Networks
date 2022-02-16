import numpy as np
import pandas as pd
import math
import sys
from matplotlib import pyplot as plt
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

def forward(X,w,af):
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
    z0 = softmax(z0)
    z.append(z0)
    return z

def backward(z,w,ytrain,si,af,lis,iter,vt,mt):
    n0 = 0.003
    l = len(z)
    p = z[l-1]-ytrain
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
    if(lis == 0):
        wg = wg-n0*g
    if(lis == 1 or lis == 2):
        vt[l-2] = 0.9*vt[l-2]+n0*g
        wg = wg-vt[l-2]
    elif(lis == 3):
        sd = np.square(g)
        vt[l-2] = 0.9*vt[l-2]+0.1*sd
        ep = math.pow(10,-8)
        sd = np.sqrt(vt[l-2]+ep)
        num = n0/sd
        wg = wg-np.multiply(num,g)
    elif(lis == 4):
        sd = np.square(g)
        mt[l-2] = 0.9*mt[l-2]+0.1*g
        vt[l-2] = 0.999*vt[l-2]+0.001*sd
        mthat = mt[l-2]/(1-math.pow(0.9,iter))
        vthat = vt[l-2]/(1-math.pow(0.999,iter))
        epsi = math.pow(10,-8)
        vthat = np.sqrt(vthat)+epsi
        num = n0/vthat
        wg = wg-np.multiply(num,mthat)
    elif(lis == 5):
        sd = np.square(g)
        mt[l-2] = 0.9*mt[l-2]+0.1*g
        vt[l-2] = 0.999*vt[l-2]+0.001*sd
        mthat = mt[l-2]/(1-math.pow(0.9,iter))
        vthat = vt[l-2]/(1-math.pow(0.999,iter))
        epsi = math.pow(10,-8)
        vthat = np.sqrt(vthat)+epsi
        num = n0/vthat
        mthat = 0.9*mthat+(0.1*g)/(1-math.pow(0.9,iter))
        wg = wg-np.multiply(num, mthat)
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
        if(lis == 0):
            wg = wg-n0*g
        if(lis == 1 or lis == 2):
            vt[i-1] = 0.9*vt[i-1]+n0*g
            wg = wg-vt[i-1]
        elif(lis == 3):
            sd = np.square(g)
            vt[i-1] = 0.9*vt[i-1]+0.1*sd
            ep = math.pow(10,-8)
            sd = np.sqrt(vt[i-1]+ep)
            num = n0/sd
            wg = wg-np.multiply(num,g)
        elif(lis == 4):
            sd = np.square(g)
            mt[i-1] = 0.9*mt[i-1]+0.1*g
            vt[i-1] = 0.999*vt[i-1]+0.001*sd
            mthat = mt[i-1]/(1-math.pow(0.9,iter))
            vthat = vt[i-1]/(1-math.pow(0.999,iter))
            epsi = math.pow(10,-8)
            vthat = np.sqrt(vthat)+epsi
            num = n0/vthat
            wg = wg-np.multiply(num,mthat)
        elif(lis == 5):
            sd = np.square(g)
            mt[i-1] = 0.9*mt[i-1]+0.1*g
            vt[i-1] = 0.999*vt[i-1]+0.001*sd
            mthat = mt[i-1]/(1-math.pow(0.9,iter))
            vthat = vt[i-1]/(1-math.pow(0.999,iter))
            epsi = math.pow(10,-8)
            vthat = np.sqrt(vthat)+epsi
            num = n0/vthat
            mthat = 0.9*mthat+(0.1*g)/(1-math.pow(0.9,iter))
            wg = wg-np.multiply(num,mthat)
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
Xtrain = Xtrain/255
ytrain = np.asarray(pd.get_dummies(ytrain))
n = Xtrain.shape[0]
np.random.seed(1)

epochs = 100
layer = [256,46]
arr = [2048]
jk = 0
af = 2
while(jk < 1):
    lis = 3
    while(lis == 3):
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
        losses = []
        xa = []
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
        while(i <= epochs and time.time()-curr < 280):
            k = arr[jk]
            j = 0
            while((j+1)*k <= n):
                X = Xtrain[j*k:(j+1)*k]
                y = ytrain[j*k:(j+1)*k]
                z = forward(X,w,af)
                if(lis == 1):
                    wy = []
                    yr = 0
                    l = len(w)
                    while(yr < l):
                        w1 = w[yr]-0.9*vt[yr]
                        wy.append(w1)
                        yr = yr+1
                    z = forward(X,wy,af)
                w = backward(z,w,y,k,af,lis,i,vt,mt)
                j = j+1
            z = forward(Xtrain,w,af)
            l = len(z)
            d = np.asarray(z[l-1])
            num = loss(d,ytrain)
            losses.append(num)
            xa.append(i)
            i = i+1
        plt.plot(xa,losses,label = str(lis))
        stri = "Architecture_22_Batchsize="+str(arr[jk])+"_Optimizer="+str(lis)
        dfk = pd.DataFrame(list(zip(xa,losses)),columns=['epochs','Loss'])
        op = output_path+stri+".csv"
        dfk.to_csv(op,index=False)
        lis = lis+1
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    stri = "Architecture_22_Batchsize="+str(arr[jk])
    plt.title(stri)
    plt.legend()
    op = output_path+stri+".png"
    plt.savefig(op)
    plt.clf()
    jk = jk+1

epochs = 100
layer = [256,46]
arr = [100,200,300,500]
jk = 0
af = 2
while(jk < 4):
    lis = 0
    while(lis <= 5):
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
        losses = []
        xa = []
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
        while(i <= epochs and time.time()-curr < 280):
            k = arr[jk]
            j = 0
            while((j+1)*k <= n):
                X = Xtrain[j*k:(j+1)*k]
                y = ytrain[j*k:(j+1)*k]
                z = forward(X,w,af)
                if(lis == 1):
                    wy = []
                    yr = 0
                    l = len(w)
                    while(yr < l):
                        w1 = w[yr]-0.9*vt[yr]
                        wy.append(w1)
                        yr = yr+1
                    z = forward(X,wy,af)
                w = backward(z,w,y,k,af,lis,i,vt,mt)
                j = j+1
            z = forward(Xtrain,w,af)
            l = len(z)
            d = np.asarray(z[l-1])
            num = loss(d,ytrain)
            losses.append(num)
            xa.append(i)
            i = i+1
        plt.plot(xa,losses,label = str(lis))
        stri = "Architecture_1_Batchsize="+str(arr[jk])+"_Optimizer="+str(lis)
        dfk = pd.DataFrame(list(zip(xa,losses)),columns=['epochs','Loss'])
        op = output_path+stri+".csv"
        dfk.to_csv(op,index=False)
        lis = lis+1
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    stri = "Architecture_1_Batchsize="+str(arr[jk])
    plt.title(stri)
    op = output_path+stri+".png"
    plt.savefig(op)
    plt.clf()
    jk = jk+1
