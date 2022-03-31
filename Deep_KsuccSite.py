from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from numpy import array
import keras

from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import pandas as pd
from IPython.display import display
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
#特征选择

from sklearn.ensemble import ExtraTreesClassifier
threshold1=0.6
threshold2=0.4

# # # # 导入数据
Mn_S = MinMaxScaler(feature_range=(0, 1))
#######
r_test_x = []
r_test_y = []
posit_1 = 0;
negat_0 = 1;
win_size = 33 # actual window size
win_size_kernel = int(win_size/2 + 1)


# define universe of possible input values
alphabet = 'ARNDCQEGHILKMFPSTWYVX'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

#
#-------------------------TEST DATASET----------------------------------------
#for positive sequence
def innertest1():
    #Input
    data = seq_record.seq
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(posit_1)
# for seq_record in SeqIO.parse("test_positive_sites.fasta", "fasta"):

for seq_record in SeqIO.parse("../Data/N16.fasta", "fasta"):
    innertest1()
print(len(r_test_x))
#for negative sequence
def innertest2():
    #Input
    data = seq_record.seq
    #print(data)
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(negat_0)
# for seq_record in SeqIO.parse("test_negative_sites.fasta", "fasta"):
for seq_record in SeqIO.parse("../Data/P16.fasta", "fasta"):
    innertest2()

# Changing to array (matrix)
r_test_x = array(r_test_x)
r_test_y = array(r_test_y)

print(r_test_x.shape)
print(r_test_y)

# ############################################################################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(r_test_x,r_test_y, test_size=0.25,random_state=666)
r_train_x = x_train
r_test_x = x_test
print(x_train.shape)
print(x_test.shape)
y_train = np.load('../Data/y_trainEGAAC.npy')
y_test =np.load('../Data/y_testEGAAC.npy')
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
##LOAD MODEL####
model = load_model('../model/model.h5')
#print("This is final ",model.layers[0].get_weights()[0][16])
r_train_y_2 = keras.utils.to_categorical(y_train, 2)
score = model.evaluate(r_train_x, r_train_y_2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
############train
Y_pred = model.predict(r_train_x)
pred0 =  model.predict_proba(r_train_x)[:,1]
# Y_pred = (Y_pred > 0.5)
# y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
# y_pred1 = np.array(y_pred1)
#
# print("Matthews Correlation : ",matthews_corrcoef(y_train, y_pred1))
# print("Confusion Matrix : \n",confusion_matrix(y_train, y_pred1))
# Acc = accuracy_score(y_train,y_pred1)
# Re = recall_score(y_train,y_pred1)
# Pre = precision_score(y_train,y_pred1)
# F1 = f1_score(y_train,y_pred1)
# MCC = matthews_corrcoef(y_train,y_pred1)
# fpr,tpr,threshold=roc_curve(y_train,pred0)
# roc_auc=auc(fpr,tpr)
# print('Acc = %.4f' % Acc)
# print('Re = %.4f' % Re)
# print('Pre = %.4f' % Pre)
# print('F1 = %.4f' % F1)
# print('MCC = %.4f' % MCC)
# print('AUC = %.4F'%roc_auc)
# # print(rf.get_params())
x_train1 = np.load('../Data/x_trainEGAAC.npy')       #1EAAC    2
x_test1  = np.load('../Data/x_testEGAAC.npy')
x_train2 = np.load('../Data/x_trainAPAAC.npy')       #1EAAC    2
x_test2  = np.load('../Data/x_testAPAAC.npy')

x_train = np.hstack((x_train1,x_train2))
x_test = np.hstack((x_test1,x_test2))
x_train=Mn_S.fit_transform(x_train)
x_test=Mn_S.transform(x_test)
print(x_train.shape)
[m,n]=x_train.shape
x_train =x_train.reshape(x_train.shape[0],n,1)
x_test=x_test.reshape(x_test.shape[0],n,1)
y_train = np.load('../Data/y_trainCTDC.npy')
y_test =np.load('../Data/y_testCTDC.npy')
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

rf = load_model('../modell/best_modelEGAAC+APAAC.h5')

predt1 = rf.predict(x_train)[:,1]
ypredt1 = np.where(predt1 >0.5 , 1, 0)
y_train = np.argmax(y_train, axis=-1)
Acc = accuracy_score(y_train,ypredt1)
print("1EGAAC+APAACt",Acc)
pred1 = rf.predict(x_test)[:,1]
ypred1 = np.where(pred1 >0.5 , 1, 0)
y_test = np.argmax(y_test, axis=-1)
Acc = accuracy_score(y_test,ypred1)
print("1EGAAC+APAAC",Acc)
# # #########################################
x_train = np.load('../Data/x_trainCTDC.npy')       #1EAAC    2
x_test  = np.load('../Data/x_testCTDC.npy')
x_train=Mn_S.fit_transform(x_train)
x_test=Mn_S.transform(x_test)
[m,n] = x_train.shape
x_train =x_train.reshape(x_train.shape[0],n,1)
x_test=x_test.reshape(x_test.shape[0],n,1)
rf = load_model('../model/best_modelCTDC.h5')
predt2 = rf.predict(x_train)[:,1]
ypredt2 = np.where(predt2 >0.5 , 1, 0)
print(ypredt2)
Acc = accuracy_score(y_train,ypredt2)
print("2CTDCt",Acc)
pred2 = rf.predict(x_test)[:,1]
ypred2 = np.where(pred2 >0.5 , 1, 0)
Acc = accuracy_score(y_test,ypred2)
print("2CTDC",Acc)
####BLUSUM62

########################整合
predt=np.vstack((predt1,predt2,pred0))
predall = predt.sum(axis=0)
predall =predall/3
y_predall = np.where(predall>=threshold1, 1, predall)
y_predall = np.where(y_predall<threshold2, 0, y_predall)
y_predall = y_predall.astype(int)
Acc = accuracy_score(y_train,y_predall)
Re = recall_score(y_train,y_predall)
Pre = precision_score(y_train,y_predall)
F1 = f1_score(y_train,y_predall)
MCC = matthews_corrcoef(y_train,y_predall)
print('Acc = %.4f' % Acc)
print('Re = %.4f' % Re)
print('Pre = %.4f' % Pre)
print('F1 = %.4f' % F1)
print('MCC = %.4f' % MCC)
fpr,tpr,threshold=roc_curve(y_train,predall)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
 lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

r_test_y_2 = keras.utils.to_categorical(y_test, 2)
Y_pred = model.predict(r_test_x)
Y_pred = (Y_pred > 0.5)
y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred1 = np.array(y_pred1)
pred0 =  model.predict_proba(r_test_x)[:,1]

pred = np.vstack((pred1,pred2,pred0))
predall = pred.sum(axis=0)
predall =predall/3
y_predall = np.where(predall>=threshold1, 1, predall)
y_predall = np.where(y_predall<threshold2, 0, y_predall)
y_predall = y_predall.astype(int)


fpr,tpr,threshold=roc_curve(y_test,predall)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
 lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

Acc = accuracy_score(y_test,y_predall)
Re = recall_score(y_test,y_predall)
Pre = precision_score(y_test,y_predall)
F1 = f1_score(y_test,y_predall)
MCC = matthews_corrcoef(y_test,y_predall)
print('Acc = %.4f' % Acc)
print('Re = %.4f' % Re)
print('Pre = %.4f' % Pre)
print('F1 = %.4f' % F1)
print('MCC = %.4f' % MCC)


