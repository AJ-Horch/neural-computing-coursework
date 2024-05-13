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


# In[114]:


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


# In[115]:


#Create inputs and targets. 
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]


# In[116]:


from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(X,y)


# In[117]:


sns.countplot(x = y, data=df)


# In[119]:


print(X.head())


# In[120]:


print(y.head())


# In[121]:


# Create split into train, val, test
# Split into train+val and test
# Stratify is being used to have an equal distribution of output classes sets
# test_size is .2, as mentioned in paper

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[122]:


# Normalize the input. Neural networks need a range of 0,1
# Use MinMaxScaler to transform features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert inputs and outputs in numpy arrays
X_train, y_train = np.array(X_train),np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[123]:


X_test.shape


# X_val and X_test, .transform was used beacause the validation and test sets should be scaled with the same parameters as the train set to avoid data leakage. fit_transform calculcates scaling values and applies. .transform only applies the calculated values. 
# 
# Cross-validation will be done in the model building phase

# # Neural Network

# ## Model parameters

# In[124]:


# create tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)


y_test = torch.tensor(y_test)
y_train = torch.tensor(y_train)


# In[125]:


print(f"Datatypes of training data: X: {X_test.dtype}, y: {y_train.dtype} ")


# In[126]:



from sklearn.model_selection import cross_val_score


# In[127]:


#implementing baisc log regression with default as test?
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

logistic_regression = linear_model.LogisticRegression()
logistic_regression_mod = logistic_regression.fit(X_train, y_train)
print(f"Baseline Logistic Regression: {round(logistic_regression_mod.score(X_test, y_test), 3)}")

pred_logistic_regression = logistic_regression_mod.predict(X_test)


# SMOTE decreased the log regression score

# #### Define architecture
# Two hidden layers, because of the universal approximation theorm. Input size is 7, output size is 3 classes
# 

# In[128]:


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


# The forward pass of the neural network takes an input tensor x and applies the fully connected layers and activation functions defined. The F.relu() function applies the ctivation function to the output of each fully connected layer. The self.dropout(x) applies dropout regularization to the output of the first and second hidden layers. Finally, the function returns the output tensor x.

# In fact, there is a theoretical finding by Lippmann in the 1987 paper “An introduction to computing with neural nets” that shows that an MLP with two hidden layers is sufficient for creating classification regions of any desired shape

# Specifically, the universal approximation theorem states that a feedforward network with a linear output layer and at least one hidden layer with any “squashing” activation function (such as the logistic sigmoid activation function) can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.
# 
# — Page 198, Deep Learning, 2016

# In[129]:


#Set up earlying stopping 
from skorch.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.0001, threshold_mode='rel', lower_is_better=True)


# In[130]:


#Multi-layer Perceptron classifier.

from skorch import NeuralNetClassifier

net = NeuralNetClassifier(
    ctgClassifier,
    lr=0.1,
    criterion = torch.nn.modules.loss.CrossEntropyLoss,   
    optimizer=torch.optim.Adam,
    callbacks=[early_stopping]
)
 


# details for tuning: https://machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/
# 

# IEEE paper mentions using cv of 10, so that is what I'll use.. 

# ### GridSearch
#  exhaustively searches through all possible combinations of hyperparameters during training the phase. Before we proceed further, we shall cover another cross-validation (CV) methods since tuning hyperparameters via grid search is usually cross-validated to avoid overfitting. Hence, For accelerating the running GridSearchCV we set: n-splits=3, n_jobs=2

# In[132]:


#Grid Search for the below parameters 
from sklearn.model_selection import GridSearchCV
params={
        'module__dropout':[0.5,0.1],
        'module__weight_constraint': [1.0, 2.0, 3.0, 4.0, 5.0],
        'lr':[0.01,0.05,0.1],
        'max_epochs':[50,100],
        'batch_size':[50,100],
        'optimizer__weight_decay':[0.01,0.5]
}
gs=GridSearchCV(net,params,cv=10, scoring=None,n_jobs=-1,verbose=0)
mlp_model=gs.fit(X_train.float(), y_train)
print(gs.best_score_,gs.best_params_)


# In[133]:


print(gs.best_score_,gs.best_params_)


# In[134]:


#mlp_model = gs.fit(X_train.float(), y_train)


# ### Prediciton

# In[135]:


from sklearn.metrics import accuracy_score


# In[136]:


#train
y_pred_train = mlp_model.predict(X_train.float())
#test
y_pred_test = mlp_model.predict(X_test.float())


# In[137]:


#getting the recall_score on train 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

print("recall_score:",recall_score(y_train, y_pred_train,average='macro'))
print("accuracy_score:",accuracy_score(y_train, y_pred_train))
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(y_train, y_pred_train)


# ## getting the recall_score on test 

# In[138]:


#getting the recall_score on test 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

print("recall_score:",recall_score(y_test, y_pred_test,average='macro'))
print("accuracy_score:",accuracy_score(y_test, y_pred_test))
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(y_test, y_pred_test)


# In[139]:


#Binarize labels in a one-vs-all fashion to use the Y_test_Roc values could be used while plotting the ROC curves 
from sklearn.preprocessing import label_binarize
Y_test_ROC = label_binarize(y_test, classes=[0, 1, 2])
print(Y_test_ROC)


# In[140]:


report = classification_report(y_test, y_pred_test)
print(report)


# In[141]:


import seaborn as sns

plt.figure(figsize=(10,6))
fx=sns.heatmap(confusion_matrix(y_test,y_pred_test), annot=True, fmt=".2f",cmap="GnBu")
fx.set_title('Confusion Matrix MLP SMOTE\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
fx.xaxis.set_ticklabels(['normal','suspect','patological'])
fx.yaxis.set_ticklabels(['normal','suspect','patological'])
plt.show()


# True negative testing

# In[142]:


normalTN_MLP = ((259+14+60+214)/993)*100
suspectTN_MLP =((307+15+248+11)/933)*100
pathTN_MLP = ((307+13+259+58)/993)*100
print("Normal TN:", normalTN_MLP)
print("Suspect TN:", suspectTN_MLP)
print("Pathlogic TN:", pathTN_MLP)


# Binarize the labels so they can be visualized using ROC/AUC

# In[143]:


from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)


# Adjust class of interest to see new line

# In[144]:


class_of_interest = 0
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id


# In[145]:


predBi = label_binarizer.transform(y_pred_test)


# In[146]:


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="blue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nNormal vs (Suspect & Pathological)")
plt.legend()
plt.show()


# In[147]:


class_of_interest = 1
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="blue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nSuspect vs (Normal & Pathological)")
plt.legend()
plt.show()


# In[148]:


class_of_interest = 2
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="blue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nPathological vs (Normal & Suspect)")
plt.legend()
plt.show()


# Classification Report: Report which includes Precision, Recall and F1-Score.
# Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
# 
# Precision = TP/TP+FP
# 
# Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.
# 
# Recall = TP/TP+FN
# 
# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

# # SVM

# In[149]:


from sklearn import svm 


# In[150]:


svmclassifier=svm.SVC(kernel='rbf')
svmclassifier.fit(X_train, y_train)
svmpred=svmclassifier.predict(X_test)


# In[151]:


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 

#param range
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear','poly'],
              'degree' : [2, 3, 4]
             }
grid = GridSearchCV(SVC(), param_grid,scoring='recall_macro', refit = True,cv=10, verbose = 0) 
  
# fitting the model for grid search 
svmModel = grid.fit(X_train,y_train) 


# In[152]:


# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 


# old best params:
# {'C': 100, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
# SVC(C=100, degree=2, gamma=0.1)

# In[153]:


grid_predictions = grid.predict(X_test) 

  
# print classification report 
print(classification_report(y_test, grid_predictions))


# In[154]:


#show confuse

plt.figure(figsize=(10,6))
fx=sns.heatmap(confusion_matrix(y_test,grid_predictions), annot=True, fmt=".2f",cmap="GnBu")
fx.set_title('Confusion Matrix SVM SMOTE \n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('True Values\n');
fx.xaxis.set_ticklabels(['normal','suspect','patological'])
fx.yaxis.set_ticklabels(['normal','suspect','patological'])
plt.show()


# In[155]:


print("recall_score:",recall_score(y_test, grid_predictions, average='macro'))
print("accuracy_score:",accuracy_score(y_test, grid_predictions))


# In[156]:


normalTN_SVM = ((324+1+331+0)/993)*100
suspectTN_SVM =((0+0+331+324)/933)*100
pathTN_SVM = ((6+7+324+324)/993)*100
print("Normal TN SVM:", normalTN_SVM)
print("Normal TN MLP:", normalTN_MLP)

print("Suspect TN SVM:", suspectTN_SVM)
print("Suspect TN MLP:", suspectTN_MLP)

print("Pathlogic TN SVM:", pathTN_SVM)
print("Pathlogic TN MLP:", pathTN_MLP)


# In[157]:


print("recall_score:",recall_score(y_test, grid_predictions,average='macro'))
print("accuracy_score:",accuracy_score(y_test, grid_predictions))
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix(y_test, grid_predictions)


# In[158]:


SVMroc = label_binarizer.transform(grid_predictions)


# In[159]:


class_of_interest = 0
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="blue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Best SVM One-vs-Rest ROC curves:\nNormal vs (Suspect & Pathological)")
plt.legend()
plt.show()


# In[160]:


class_of_interest = 1
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="red",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Best SVM One-vs-Rest ROC curves:\nSuspect vs (Normal & Pathological)")
plt.legend()
plt.show()


# In[161]:


class_of_interest = 2
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    predBi[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="orange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM One-vs-Rest ROC curves:\nPathological vs (Normal & Suspect)")
plt.legend()
plt.show()


# ## Export best model

# In[162]:


get_ipython().system('pip install joblib')


# In[163]:


bestSVM=grid_predictions
import joblib


# In[164]:


#save model to disk
filename = 'best_svm_model.joblib'
joblib.dump(bestSVM, filename)


# In[165]:


# load the model from disk
svmload=joblib.load('best_svm_model.joblib')


# In[166]:


loadpredict = svmload.predict(X_test)


# In[ ]:




