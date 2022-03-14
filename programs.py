#codiing: utf-8
#Yuuki Imagawa 
import sys,os 
import numpy as np 
import argparse 
import random
from matplotlib import pyplot as pyp 
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split  

def split_data(x,y,N,N_train,N_test):
    xy=np.concatenate([x,y],1)
    np.random.shuffle(xy)
    
    #split train_data&test_data 
    xy_train, xy_test=np.split(xy,[int(N*0.9)])
    xy_train_positive_x0=[]
    xy_train_positive_x1=[]
    xy_train_positive_x2=[]
    xy_train_negative_x0=[]
    xy_train_negative_x1=[]
    xy_train_negative_x2=[]
    i=0
    for i in range(N_train):
        if (xy_train[i,2]==1):
           xy_train_positive_x0.append(xy_train[i,0])
           xy_train_positive_x1.append(xy_train[i,1])
           xy_train_positive_x2.append(xy_train[i,2])
        if (xy_train[i,2]==-1):
           xy_train_negative_x0.append(xy_train[i,0])
           xy_train_negative_x1.append(xy_train[i,1])
           xy_train_negative_x2.append(xy_train[i,2])
 
    xy_train_positive_x0=np.array(xy_train_positive_x0)
    xy_train_positive_x0=xy_train_positive_x0.reshape(xy_train_positive_x0.size,1)
    xy_train_positive_x1=np.array(xy_train_positive_x1)
    xy_train_positive_x1=xy_train_positive_x1.reshape(xy_train_positive_x1.size,1)
    xy_train_positive_x2=np.array(xy_train_positive_x2)
    xy_train_positive_x2=xy_train_positive_x2.reshape(xy_train_positive_x2.size,1)
    xy_train_positive=np.concatenate([xy_train_positive_x0,xy_train_positive_x1,xy_train_positive_x2],1)

    xy_train_negative_x0=np.array(xy_train_negative_x0)
    xy_train_negative_x0=xy_train_negative_x0.reshape(xy_train_negative_x0.size,1)
    xy_train_negative_x1=np.array(xy_train_negative_x1)
    xy_train_negative_x1=xy_train_negative_x1.reshape(xy_train_negative_x1.size,1)
    xy_train_negative_x2=np.array(xy_train_negative_x2)
    xy_train_negative_x2=xy_train_negative_x2.reshape(xy_train_negative_x2.size,1)
    xy_train_negative=np.concatenate([xy_train_negative_x0,xy_train_negative_x1,xy_train_negative_x2],1)
    
    xy_test_positive_x0=[]
    xy_test_positive_x1=[]
    xy_test_positive_x2=[]
    xy_test_negative_x0=[]
    xy_test_negative_x1=[]
    xy_test_negative_x2=[]
    i=0
    for i in range(N_test):
        if (xy_test[i,2]==1):
           xy_test_positive_x0.append(xy_test[i,0])
           xy_test_positive_x1.append(xy_test[i,1])
           xy_test_positive_x2.append(xy_test[i,2])
        if (xy_test[i,2]==-1):
           xy_test_negative_x0.append(xy_test[i,0])
           xy_test_negative_x1.append(xy_test[i,1])
           xy_test_negative_x2.append(xy_test[i,2])
  
    xy_test_positive_x0=np.array(xy_test_positive_x0) 
    xy_test_positive_x0=xy_test_positive_x0.reshape(xy_test_positive_x0.size,1)
    xy_test_positive_x1=np.array(xy_test_positive_x1)
    xy_test_positive_x1=xy_test_positive_x1.reshape(xy_test_positive_x1.size,1)
    xy_test_positive_x2=np.array(xy_test_positive_x2)
    xy_test_positive_x2=xy_test_positive_x2.reshape(xy_test_positive_x2.size,1)
    xy_test_positive=np.concatenate([xy_test_positive_x0,xy_test_positive_x1,xy_test_positive_x2],1)
    
    xy_test_negative_x0=np.array(xy_test_negative_x0)
    xy_test_negative_x0=xy_test_negative_x0.reshape(xy_test_negative_x0.size,1)
    xy_test_negative_x1=np.array(xy_test_negative_x1)
    xy_test_negative_x1=xy_test_negative_x1.reshape(xy_test_negative_x1.size,1)
    xy_test_negative_x2=np.array(xy_test_negative_x2)
    xy_test_negative_x2=xy_test_negative_x2.reshape(xy_test_negative_x2.size,1)
    xy_test_negative=np.concatenate([xy_test_negative_x0,xy_test_negative_x1,xy_test_negative_x2],1)

    np.savetxt("train.txt",xy_train)
    np.savetxt("_test.txt",xy_test)
    np.savetxt("train_positive.txt",xy_train_positive)
    np.savetxt("train_negative.txt",xy_train_negative)
    np.savetxt("test_positive.txt",xy_test_positive)
    np.savetxt("test_negative.txt",xy_test_negative)
 
    x_train1,x_train2,y_train=np.hsplit(xy_train,3)
    x_train=np.concatenate([x_train1,x_train2],1)
    x_test1,x_test2,y_test=np.hsplit(xy_test,3)
    x_test=np.concatenate([x_test1,x_test2],1)
    train_data=np.concatenate([x_train,y_train],1)
    test_data=np.concatenate([x_test,y_test],1) 
    return x_train,x_test,y_train,y_test



def make_kernel_matrix(x_train,γ,N_train): 
    x_train=x_train.T
    x2=np.power(x_train,2)
    Z=np.sum(x2,axis=0) 
    Z=Z.reshape(N_train,1) 
    A=np.tile(Z,(1,N_train)) 

    Z=Z.reshape(1,N_train) 
    B=np.tile(Z,(N_train,1)) 
    D=np.dot(x_train.T,x_train)  
    K=A+B-2*D 
    K=np.exp(-γ*K) 
    return K

def select_s_t(I_up,I_low,U_up,U_low,N_train):
    s=np.argmax(U_up)
    s=I_up[s]
    t=np.argmin(U_low)
    t=I_low[t]
    while s==t:
      t=random.randint(0,N_train-1)
    return s,t


def estimate_delta_beta(y_train,C,s,t,beta,K):  
    kernel_s_column=K[:,s] 
    ds=y_train[s]-np.dot(beta.T,kernel_s_column)
    kernel_t_column=K[:,t]
    dt=y_train[t]-np.dot(beta.T,kernel_t_column)
    delta_beta=(ds-dt)/(K[s,s]-2.0*K[s,t]+K[t,t])

    if (y_train[s]==1 and y_train[t]==1): 
        L=max(-beta[s],beta[t]-C) 
        u=min(C-beta[s],beta[t]) 
    if (y_train[s]==1 and y_train[t]==-1): 
        L=max(beta[t],-beta[s]) 
        u=min(beta[t]+C,C-beta[s])
    if (y_train[s]==-1 and y_train[t]==1): 
        L=max(beta[t]-C,-C-beta[s])
        u=min(beta[t],-beta[s])
    if (y_train[s]==-1 and y_train[t]==-1):
        L=max(beta[t],-C-beta[s]) 
        u=min(-beta[s],beta[t]+C) 
    if (L>delta_beta):
        delta_beta=L
    if (u<delta_beta):
        delta_beta=u
    return delta_beta,ds,dt


def update_omega(y_train,beta,K,N_train):
    y_train=y_train.reshape(1,N_train)
    beta=beta.reshape(1,N_train)

    tmp1=-np.dot(y_train,beta.T)
    tmp2=np.dot(beta,K)
    tmp3=np.dot(tmp2,beta.T)
    tmp1=float(tmp1[0])
    tmp3=float(tmp3[0])
    omega=0.5*tmp3+tmp1
    return omega


def update_U(K,delta_beta,U,N_train,s,t):
    j=0
    for j in range(N_train):
        U=U.reshape(1,N_train)
        tmp4=K[j,s]-K[j,t]
        U[0,j]=U[0,j]-delta_beta*tmp4
    U=U.flatten()
    return U

def update_alpha(N_train,alpha,beta,y_train):
    for i in range(N_train):
        alpha[i]=beta[i]*y_train[i]
    return alpha

def update_I_up_I_low(alpha,C,N_train,y_train):
    I_up=[]
    I_low=[]
    i=0
    for i in range(N_train):
        if (alpha[i]>0 and alpha[i]<C):
            I_up.append(i)
            I_low.append(i)
        if (alpha[i]==0 and y_train[i]==1):
            I_up.append(i)
        if (alpha[i]==C and y_train[i]==-1):
            I_up.append(i)
        if (alpha[i]==0 and y_train[i]==-1):
            I_low.append(i)
        if (alpha[i]==C and y_train[i]==1):
            I_low.append(i)
    return I_up,I_low


def update_U_up_U_low(N,I_up,I_low,U):
    U_up=np.zeros(len(I_up))
    U_low=np.zeros(len(I_low))
    I_up=np.array(I_up)
    I_low=np.array(I_low)
    i=0
    for i in range(len(I_up)):
        U_up[i]=U[I_up[i]]
    i=0
    for i in range(len(I_low)):
        U_low[i]=U[I_low[i]]
    return U_up,U_low


def make_omega_graph(Nitr,omega_array):
    Iitr=np.zeros(Nitr)
    i=0
    for i in range(Nitr):
        Iitr[i]=i

    pyp.title("omega-graph",{"fontsize":25})
    pyp.xlabel("Nitr",{"fontsize":15})
    pyp.ylabel("omega",{"fontsize":15})
    pyp.plot(Iitr,omega_array)
    pyp.show()
    return Iitr 


def estimate_alpha(alpha,y_train,beta,N_train):
    for i in range(N_train):
        alpha[i]=y_train[i]*beta[i]
    return alpha

def estimate_I_for_b(I,alpha,C,N_train,I_for_b):
    I=I.reshape(1,N_train)
    I=np.array(I)
    i=0
    for i in range(N_train):
        if (alpha[i]>0 and alpha[i]<C):
            I_for_b.append(i)
    I_for_b=np.array(I_for_b)
    I_for_b=I_for_b.reshape(1,I_for_b.size)
    return I_for_b

def estimate_U_for_b(N_train,I_for_b,U,U_for_b):
    I_for_b=np.array(I_for_b)
    U_for_b=np.zeros(I_for_b.size)
    l=0
    for l in range(I_for_b.size):
        U_for_b[l]=U[I_for_b[l]]
    return U_for_b

def estimate_b(U_for_b):
    b=sum(U_for_b)/len(U_for_b)
    return b

def estimate_fx(x_train,N_train,γ,alpha,y_train,b):
    Ngrid=500 
    x_train=x_train.T
    x2=np.power(x_train,2)
    Z=np.sum(x2,axis=0)
    Z=Z.reshape(1,N_train)
    A=np.tile(Z,(Ngrid**2,1))

    X0=np.linspace(-20,20,Ngrid)
    X1=np.linspace(-25,15,Ngrid)
    X0=X0.reshape(1,Ngrid)
    X1=X1.reshape(1,Ngrid)
    Xgrid=np.zeros((2,Ngrid**2))  
    for i in range(Ngrid):
        for j in range(Ngrid): 
            k = j+i*Ngrid 
            Xgrid[0,k]=X0[0,i] 
            Xgrid[1,k]=X1[0,j] 

    Xgrid2=np.power(Xgrid,2)
    Zgrid=np.sum(Xgrid2,axis=0)
    Zgrid=Zgrid.reshape(1,Ngrid**2)
    B=np.tile(Zgrid,(N_train,1))
    B=B.T
    D=np.dot(Xgrid.T,x_train)
    E=A+B-2.0*D 
    Kernel=np.exp(-γ*E)
    
    beta=np.zeros(N_train)
    i=0
    for i in range(N_train):
        beta[i]=alpha[i]*y_train[i]
    beta=beta.reshape(1,N_train)
    F=np.dot(beta,Kernel.T)
    fx=F+b
    return fx,X0,X1,Ngrid

def estimate_fx_test(beta,x_train,x_test,N_train,N_test,γ,b):
    x_train=x_train.T
    x2=np.power(x_train,2)
    Z=np.sum(x2,axis=0)
    Z=Z.reshape(N_train,1)
    A=np.tile(Z,(1,N_train))
    B=Z.T
    D=np.dot(x_train.T,x_train)
    K=A+B-2*D
    Kernel=np.exp(-γ*K)
    beta=beta.reshape(1,N_train)
    F=np.dot(beta,Kernel)
    fx_train=F+b
    return fx_train

def estimate_f_value(N_train,y_train,fx_train):
    #TP
    TP=[]
    TN=[]
    FP=[]
    FN=[]
    i=0
    for i in range(N_train):
        tmp=y_train[i]*fx_train[0,i]
        if (tmp>=0 and y_train[i]==1):
           TP.append(tmp)
        if (tmp>=0 and y_train[i]==-1):
           TN.append(tmp)
        if (tmp<0 and y_train[i]==1):
           FP.append(tmp)
        if (tmp<0 and y_train[i]==-1):
           FN.append(tmp)
    Accuracy=(len(TP)+len(TN))/N_train
    Precision=(len(TP))/(len(TP)+len(FP))
    if(TN==0 and FP==0):
        Recall=0
    else: 
        Recall=(len(TP))/(len(TP)+len(FN))
    F_measure=(2*Precision*Recall)/(Precision+Recall)
    Specificity=(len(TN))/(len(TN)+len(FP))
    False_Positive_Rate=(len(FP))/(len(TN)+len(FP))




#def write_test_data(N_test,x_test,y_test):
    #x_test=x_test.reshape(2,N_test)
    #y_test=y_test.reshape(1,N_test)
    #x_test_y_test=np.zeros((N_test,3))
    #for i in range(N_test):
        #x_test_y_test[i,0]=x_test[0,i]
        #x_test_y_test[i,1]=x_test[1,i]
        #x_test_y_test[i,2]=y_test[0,i]
    #np.savetxt('x_test_y_test.txt',x_test_y_test)

def wrt_split_test_positive_negtive_data(x_test,y_test,N_test):
    xy=np.concatenate([x_test,y_test],1)
    positive_data=[]
    negative_data=[]
    i=0
    for i in range(N_test):
        tmp=xy[i,2]
        if (tmp==1):
          positive_data.append(xy[i])
        if (tmp==-1):
          negative_data.append(xy[i])
    positive_data=np.array(positive_data)
    negative_data=np.array(negative_data)
    np.savetxt('positive_test.txt',positive_data)
    np.savetxt('negative_test.txt',negative_data)

def write_result(X0,X1,fx,Ngrid):
    file=open('result.gnu', 'w')
    for i in range(Ngrid):
        for j in  range(Ngrid):
            k=j+i*Ngrid
            file.write(str(X0[0,i]))  
            file.write('\t')
            file.write(str(X1[0,j]))  
            file.write('\t')
            file.write(str(fx[0,k]))
            file.write('\n')
        file.write('\n') 
    file.close()        
    
 
