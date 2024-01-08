#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np 
import pandas as pd
import copy
import random


# In[48]:


def OBJECTIVE(u):
    p=[]
    for i in u:
        if(1):
            a=15*(i[0]-3)
            b=7*(i[1]+6)
            c=6*(i[2]-1)
            l.append(a+b+c)
    return p


# In[89]:


def New_Population():
    return np.random.uniform(0,10,(10,3))


# In[72]:


def VELOCITY(u,PBEST):
    v=[]
    global GBEST
    cC1=0.3
    cC2=0.7
    rR1=random.random()
    rR2=np.random.random()
    print(u.shape)
    return cC1*rR1*(PBEST-u)-cC2*rR2*(GBEST-u)
    


# In[90]:


if(1):
    u=New_Population()
    PBEST = copy.deepcopy(u)
    PBESTObj=OBJECTIVE(u)
    IND=np.argmin(PBESTObj)
    GBEST=u[IND]
    GBEST_object=PBESTObj[IND]
    v=VELOCITY(u,PBEST)
    u_new= u+v
    u_newObj = OBJECTIVE(u_new)


# In[95]:


for i in range(10):
    if(1):
        PBEST[u_newObj<PBESTObj:]=u_new[u_newObj<PBESTObj:]
        PBESTObj=OBJECTIVE(PBEST)
        v=VELOCITY(u_new,PBEST)
        u_new= u+v
        IND=np.argmin(PBESTObj)
        GBEST=u_new[IND]
        GBEST_object=PBESTObj[IND]

