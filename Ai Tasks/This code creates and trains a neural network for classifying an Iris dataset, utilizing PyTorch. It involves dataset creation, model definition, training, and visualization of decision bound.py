#!/usr/bin/env python
# coding: utf-8

# ## DataSet creation

# In[2]:


get_ipython().system('pip install torch')


# In[64]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris['data']
Y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42) 


# In[65]:


from sklearn.datasets import make_classification
X, Y = make_classification(
    n_features=4, n_redundant=0, n_informative=3, n_clusters_per_class=2, n_classes=3
)


# In[ ]:





# ## Step 2: Split the data and make tensor

# In[66]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[67]:


import torch
import numpy as np

X_trainTensor = torch.from_numpy(X_train) # convert to tensors
y_trainTensor = torch.from_numpy(y_train)
X_testTensor = torch.from_numpy(X_test)
y_testTensor = torch.from_numpy(y_test)


# ## Built Neural network

# In[68]:


from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self):
        self.X=torch.from_numpy(X_train)
        self.Y=torch.from_numpy(Y_train)
        self.len=self.X.shape[0]
    def __getitem__(self,index):      
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.len
    
    
data=Data()
print(data.X[0] , data.Y[0])
print(data. __getitem__(0))
loader=DataLoader(dataset=data,batch_size=64)


# In[69]:


import torch.nn as nn

#First of all, we will define the dimensions of the network.

input_dim=4     # how many Variables are in the dataset
hidden_dim = 50 # hidden layers
output_dim=3    # number of classe


class Net(nn.Module):
    def __init__(self,input,H,output):
        super(Net,self).__init__()
        self.linear1=nn.Linear(input,H)
        self.linear2=nn.Linear(H,output)
 
        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x
    
clf=Net(input_dim,hidden_dim,output_dim)    # object of the network
print(clf.parameters)                       # object parameter 


# ## Model Training

# In[70]:


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(clf.parameters(), lr=0.2)

# After defining the criterion and optimizer, we are ready to train our model. Using the following lines of codes we can train it.

learning_rate = 1e-1
loss_list = []
for t in range(3000):
    y_pred = clf(X_trainTensor.float())
    loss = criterion(y_pred, y_trainTensor.long()) # calculate the loss
    loss_list.append(loss.item())
    optimizer.zero_grad() # Venish the gradient before the next step to save the memory
    loss.backward()  # Backward of the network 
    optimizer.step() # update the parameters


# In[71]:


import matplotlib.pyplot as plt
step = np.linspace(0,3000,3000)
plt.plot(step,np.array(loss_list))


# In[72]:


params = list(clf.parameters())
w = params[0].detach().numpy()[0]
b = params[1].detach().numpy()[0]
t= params[3].detach().numpy()[0]
plt.scatter(X[:, 0], X[:, 1], c=Y,cmap='jet')
u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
plt.plot(u, (0.5-b-w[0]*u)/w[1])
plt.plot(u, (0.5-t-w[0]*u)/w[1])
plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)


# In[ ]:





# In[ ]:




