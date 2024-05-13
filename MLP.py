#!/usr/bin/env python
# coding: utf-8

# # import and preprocess

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


df = pd.read_excel('ctgdata.xlsx')


# In[3]:


df = df[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'SUSP', 'CLASS', 'NSP']]


# In[5]:


# Using KBEst Algo with f_classif to perform ANOVA which:
#determines the degree of linear dependency between the target variable and features.
from sklearn.feature_selection import SelectKBest #Feature Selector
from sklearn.feature_selection import f_classif #ANOVA


# In[8]:


# borrowed from: https://www.kaggle.com/code/christopherwsmith/fetal-health-a-quick-guide-to-high-accuracy
def Plotter(plot, x_label, y_label, x_rot=None, y_rot=None,  fontsize=12, fontweight=None, legend=None, save=False,save_name=None):
    """
    Helper function to make a quick consistent plot with few easy changes for aesthetics.
    Input:
    plot: sns or matplot plotting function
    x_label: x_label as string
    y_label: y_label as string
    x_rot: x-tick rotation, default=None, can be int 0-360
    y_rot: y-tick rotation, default=None, can be int 0-360
    fontsize: size of plot font on axis, defaul=12, can be int/float
    fontweight: Adding character to font, default=None, can be 'bold'
    legend: Choice of including legend, default=None, bool, True:False
    save: Saves image output, default=False, bool
    save_name: Name of output image file as .png. Requires Save to be True.
               default=None, string: 'Insert Name.png'
    Output: A customized plot based on given parameters and an output file
    
    """
    #Ticks
    ax.tick_params(direction='out', length=5, width=3, colors='k',
               grid_color='k', grid_alpha=1,grid_linewidth=2)
    plt.xticks(fontsize=fontsize, fontweight=fontweight, rotation=x_rot)
    plt.yticks(fontsize=fontsize, fontweight=fontweight, rotation=y_rot)

    #Legend
    if legend==None:
        pass
    elif legend==True:
        
        plt.legend()
        ax.legend()
        pass
    else:
        ax.legend().remove()
        
    #Labels
    plt.xlabel(x_label, fontsize=fontsize, fontweight=fontweight, color='k')
    plt.ylabel(y_label, fontsize=fontsize, fontweight=fontweight, color='k')

    #Removing Spines and setting up remianing, preset prior to use.
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(3)
    
    if save==True:
        plt.savefig(save_name)


# In[9]:


#Feature Selection
X=df.drop(['NSP'], axis=1)
Y=df['NSP']
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns

#Visualize the feature scores
fig, ax=plt.subplots(figsize=(7,7))
plot=sns.barplot(data=featureScores, x='Score', y='Feature', palette='viridis',linewidth=0.5, saturation=2, orient='h')
Plotter(plot, 'Score', 'Feature', legend=False, save=True, save_name='Feature Importance.png')#Plotter function for aesthetics
plot


# In[11]:


#Selection method
selection=featureScores[featureScores['Score']>=200]#Selects features that scored more than 200
selection=list(selection['Feature'])#Generates the features into a list
selection.append('NSP')#Adding the Level string to be used to make new data frame
df_feat=df[selection] #New dataframe with selected features
df_feat = df_feat.drop(columns=['SUSP'])
df_feat.head() #Lets take a look at the first 5


# In[12]:


# make things simple
data = df_feat
# Encoding the output. labels need to go from 0-2 in order to work with tensor
# 0 = Normal, 1 = Suspect, 2 = Pathologic
# borrowed from https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab

class2idx = {
    1:0,
    2:1,
    3:2
}

idx2class = {v: k for k, v in class2idx.items()}
data['NSP'].replace(class2idx, inplace=True)


# In[13]:


#Create inputs and targets. 
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[15]:


# Normalize the input. Neural networks need a range of 0,1
# Use MinMaxScaler to transform features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert inputs and outputs in numpy arrays
X_train, y_train = np.array(X_train),np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[16]:


# create tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)


y_test = torch.tensor(y_test)
y_train = torch.tensor(y_train)


# # MLP

# In[17]:


import torch.nn.functional as F
import torch.nn as nn

class ctgClassifier(nn.Module):
    def __init__(self, dropout=0.5, weight_constraint=1.0):        
        super(ctgClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.layer_1 = nn.Linear(7, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 3) 
        
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)      
        x = F.relu(self.layer_2(x))
        x = self.dropout(x)
        return x 


# In[30]:


#Set up earlying stopping 
from skorch.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.0001, threshold_mode='rel', lower_is_better=True)


# In[32]:


#Multi-layer Perceptron classifier.

from skorch import NeuralNetClassifier

net = NeuralNetClassifier(
    ctgClassifier,
    lr=0.1,
    criterion = torch.nn.modules.loss.CrossEntropyLoss,   
    optimizer=torch.optim.Adam,
    callbacks=[early_stopping],
    optimizer__weight_decay==[0.01],
    module__dropout==[0.1],
    module__weight_constraint==[2.0],
    max_epochs==[100],
    batch_size==[100],
)
 


# In[26]:


mlp_model = net
#test
y_pred_test = mlp_model.predict(X_test.float())


# In[ ]:


#getting the recall_score on test 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

print("recall_score:",recall_score(y_test, y_pred_test,average='macro'))
print("accuracy_score:",accuracy_score(y_test, y_pred_test))
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(y_test, y_pred_test)


# In[ ]:


inarize labels in a one-vs-all fashion to use the Y_test_Roc values could be used while plotting the ROC curves 
from sklearn.preprocessing import label_binarize
Y_test_ROC = label_binarize(y_test, classes=[0, 1, 2])
print(Y_test_ROC)


# In[ ]:


report = classification_report(y_test, y_pred_test)
print(report)


# In[ ]:


import seaborn as sns

plt.figure(figsize=(10,6))
fx=sns.heatmap(confusion_matrix(y_test,y_pred_test), annot=True, fmt=".2f",cmap="GnBu")
fx.set_title('Confusion Matrix MLP\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
fx.xaxis.set_ticklabels(['normal','suspect','patological'])
fx.yaxis.set_ticklabels(['normal','suspect','patological'])
plt.show()


# In[ ]:




