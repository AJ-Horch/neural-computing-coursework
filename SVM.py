#!/usr/bin/env python
# coding: utf-8

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
df.head()


# ## Data exploration
# I know from the initial data exploration done during the group coursework that this dataset has been preprocessed. However during the group cw we decided against any feature extraction/transfromation because it was not necessary. Some research has suggested feature extraction increases accuracy. Thus, features with low correlations and vriances will be removed. 

# In[3]:


#class distribution
sns.countplot(x = 'NSP', data=df)


# In[4]:


df = df[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'SUSP', 'CLASS', 'NSP']]


# UCI says there are only 23 attributes, but this shows 35. According to UCI feature are: LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, NMax, Nzeros, Mode, Mean, Median, Variance, Tendency, Class, NSP.I'll keep SUSP because it appears to have a stornger corr than most. All others will be dropped

# In[5]:


#reduced to 23 attributes. 
df.head()


# the data is imbalanced, which was known. SMOTE did little to improve this during prelimbary scikit phase. Will feature extraction improve?
# /

# In[6]:


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


# In[7]:


fig, ax=plt.subplots(figsize=(20,20))#Required outside of function. This needs to be activated first when plotting in every code block
cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)
plot=sns.heatmap(df.corr(),annot=True, cmap=cmap, linewidths=1)
Plotter(plot, None, None, 90,legend=False, save=True, save_name='Corr.png')


# Shades of red are more correlated than blue. Looking at the NSP col/row it appears that LB, DS, DP, ASTV, ALTV, Varaince, SUSP, CLASS have the best correlation. 

# In[8]:


# Using KBEst Algo with f_classif to perform ANOVA which:
#determines the degree of linear dependency between the target variable and features.
from sklearn.feature_selection import SelectKBest #Feature Selector
from sklearn.feature_selection import f_classif #ANOVA


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


# SUSP isnt listed in the offical attribute information, but it was correlated and now shows it has the highet linear dependency. I'm going to exclude it because its not listed and could be an outlier. 250 looks to be a good cut off point for feature selection. 

# In[10]:


#Selection method
selection=featureScores[featureScores['Score']>=200]#Selects features that scored more than 200
selection=list(selection['Feature'])#Generates the features into a list
selection.append('NSP')#Adding the Level string to be used to make new data frame
df_feat=df[selection] #New dataframe with selected features
df_feat = df_feat.drop(columns=['SUSP'])
df_feat.head() #Lets take a look at the first 5


# In[11]:


# borrowed from: borrowed from: https://www.kaggle.com/code/christopherwsmith/fetal-health-a-quick-guide-to-high-accuracy
sns.pairplot(df_feat, hue='NSP')


# IEEE paper mentions 7 features so this seems to be a good choice!
# 
# Classes 2 and 3 are diffcult to distingish here. 

# ## Splitting, sclaing, encoding

# In[12]:


# make things simple
data = df_feat


# In[13]:


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


# In[14]:


#Create inputs and targets. 
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]


# In[15]:


from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(X,y)


# In[16]:


sns.countplot(x = y, data=df)


# In[17]:


print(X.head())


# In[18]:


print(y.head())


# In[19]:


# Create split into train, val, test
# Split into train+val and test
# Stratify is being used to have an equal distribution of output classes sets
# test_size is .2, as mentioned in paper

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[20]:


# Normalize the input. Neural networks need a range of 0,1
# Use MinMaxScaler to transform features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert inputs and outputs in numpy arrays
X_train, y_train = np.array(X_train),np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[21]:


X_test.shape


# X_val and X_test, .transform was used beacause the validation and test sets should be scaled with the same parameters as the train set to avoid data leakage. fit_transform calculcates scaling values and applies. .transform only applies the calculated values. 
# 
# Cross-validation will be done in the model building phase

# # Neural Network

# ## Model parameters

# In[22]:


# create tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)


y_test = torch.tensor(y_test)
y_train = torch.tensor(y_train)


# In[23]:


print(f"Datatypes of training data: X: {X_test.dtype}, y: {y_train.dtype} ")


# In[24]:



from sklearn.model_selection import cross_val_score


# # SVM

# In[25]:


from sklearn import svm 


# In[26]:


svmclassifier=svm.SVC(kernel='rbf')
svmclassifier.fit(X_train, y_train)
svmpred=svmclassifier.predict(X_test)


# In[36]:



from sklearn.svm import SVC 

SVM_model = SVC(C=1000,degree=2,gamma=1)
SVM = SVM_model.fit(X_test,y_test) 


# In[38]:


svmtest = SVM.predict(X_test)

#getting the recall_score on test 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
print("recall_score:",recall_score(y_test, svmtest,average='macro'))
print("accuracy_score:",accuracy_score(y_test, svmtest))
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(y_test, svmtest)


# In[30]:


get_ipython().system('pip install joblib')


# In[31]:


bestSVM=grid_predictions
import joblib


# In[ ]:


#save model to disk
filename = 'best_svm_model.joblib'
joblib.dump(bestSVM, filename)


# In[ ]:


# load the model from disk
svmload=joblib.load('best_svm_model.joblib')


# In[ ]:


loadpredict = svmload.predict(X_test)


# In[ ]:





# In[ ]:




