#codiing: utf-8
#(c) Yuuki Imagawa
import sys
import os
import numpy as np
import argparse
import random
import copy
from data import read_data
from matplotlib import pyplot as pyp
from mpl_toolkits.mplot3d import Axes3D
from programs import * 
parser=argparse.ArgumentParser()
parser.add_argument('--d',type=int,default=2,help='dimention of input x (default=2)')
parser.add_argument('--C',type=float,default=10.0,help='regurality parameter (default=10.0)')
parser.add_argument('--Nitr',type=int,default=1000,help='Total number of iteration step (default=1000)')
parser.add_argument('--γ',type=float,default=0.01,help='regurality parameter (default=1000)')
args = parser.parse_args()
d=args.d
C=args.C
Nitr=args.Nitr
γ=args.γ
N=200
N_train=int(N*0.9)
N_test=int(N*0.1)

#read data 
(x,y)=read_data(d)
x=x.reshape(N,2)
y=y.reshape(N,1)


#split_data
(x_train,x_test,y_train,y_test)=split_data(x,y,N,N_train,N_test)

#make_kernel_matrix
(K)=make_kernel_matrix(x_train,γ,N_train)

beta=np.zeros(N_train)
alpha=np.zeros(N_train)

I=[]
for i in range(N_train):
        I.append(i)

I=np.array(I)
U=y_train.copy()
I_for_b=[]
U_for_b=[]
b_for_b=[]


s=random.randint(0,N_train-1)
t=random.randint(0,N_train-1)
while s==t:
   t=random.randint(0,N_train-1)

delta_beta_array=np.zeros(Nitr)

omega_array=np.zeros(Nitr)
i=0
while i<Nitr:
        (delta_beta,ds,dt)=estimate_delta_beta(y_train,C,s,t,beta,K)
        beta[s]=beta[s]+delta_beta
        beta[t]=beta[t]-delta_beta
        (omega)=update_omega(y_train,beta,K,N_train)
        omega_array[i]=omega
        print(i,omega_array[i])
        (U)=update_U(K,delta_beta,U,N_train,s,t)
        (alpha)=update_alpha(N_train,alpha,beta,y_train)
        (I_up,I_low)=update_I_up_I_low(alpha,C,N_train,y_train)
        (U_up,U_low)=update_U_up_U_low(N_train,I_up,I_low,U)
        #(s,t)=select_s_t(I_up,I_low,U_up,U_low,N_train)
        s=random.randint(0,N_train-1)
        t=random.randint(0,N_train-1)
        while s==t:
           t=random.randint(0,N_train-1)
        i=i+1

#make_omega_graph
#(Iitr)=make_omega_graph(Nitr,omega_array)

#estimate_alpha
alpha=estimate_alpha(alpha,y_train,beta,N_train)

#estimate_U
estimate_I_for_b(I,alpha,C,N_train,I_for_b)
U_for_b=estimate_U_for_b(N_train,I_for_b,U,U_for_b)

#estimate_b
b=estimate_b(U_for_b)

#estimate_fx
(fx,X0,X1,Ngrid)=estimate_fx(x_train,N_train,γ,alpha,y_train,b)

#estimate_fx_test
(fx_train)=estimate_fx_test(beta,x_train,x_test,N_train,N_test,γ,b)

#write_test_data
#write_test_data(N_test,x_test,y_test)

#wrt_split_test_positive_negtive_data
wrt_split_test_positive_negtive_data(x_test,y_test,N_test)

#estimate_f_value
estimate_f_value(N_train,y_train,fx_train)

#write_result
write_result(X0,X1,fx,Ngrid) 
