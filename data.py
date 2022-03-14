#coding: utf-8
#(c) Yuuki Imagawa
import sys
import os
import numpy as np

def read_data(d):  
    f=open('train.csv','r')
    line=f.readlines()
    f.close()
    tmp=np.loadtxt(line)
    N=tmp.shape[0] 
    x=np.zeros((N,d))
    y=np.zeros(N)
    for i in range(N): 
        x[i,0]=tmp[i,0]
        x[i,1]=tmp[i,1]
        y[i]=tmp[i,2] 
    return x,y 
