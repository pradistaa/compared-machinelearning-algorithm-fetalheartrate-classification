# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:33:42 2020

@author: Admin
"""


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def prep(dataset):
    #dataset = dataset.drop(["FileName","Date","SegFile","b","e","DR"],axis=1)
    print("========= Head =========")
    print(dataset.head())
    print("========= Column =========")
    print(dataset.columns)
    print("========= Shape =========")
    print(dataset.shape)
    print("========= Missing Values =========")
    print(dataset.isnull().sum())
    dataset = dataset.dropna() 
    print("========= Data Type =========")
    print(dataset.dtypes)
    return dataset

def preproses_minmax(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(data)
    return rescaledX
    
    
def preproses_standarize(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data)
    rescaledX = scaler.transform(data)
    return rescaledX

def holdout(X,y,test):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=0)
    return X_train, X_test, y_train, y_test

def klasifikasi_kfold(X,y):
    
    run_kfold = pd.DataFrame({},columns=['Model','Fold','Akurasi'])  

    def hasil(expected,predicted,nama,i):
        print("========== {} - Fold - {} ==========".format(nama,i))
        print(metrics.confusion_matrix(expected,predicted))
        print(metrics.classification_report(expected,predicted))
        print("Accuracy = {}".format(accuracy_score(expected, predicted)))
        
    i = 1
    skf = StratifiedKFold(n_splits=3)
    for train,test in skf.split(X,y):
        X_train, X_test, y_train, y_test = X[train],X[test],y[train],y[test]
    
        #Logistic Regression
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        hasil(y_test,model.predict(X_test),"Logistic Regression",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"Logistic Regression",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #Decision Tree
        model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
        hasil(y_test,model.predict(X_test),"Decision Tree",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"Decision Tree",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #MLPClassifier
        model = MLPClassifier(activation='logistic', solver='lbfgs', alpha = 1e-5, 
                            hidden_layer_sizes=(16,14), random_state=0,max_iter=100000)
        model.fit(X_train, y_train)
        hasil(y_test,model.predict(X_test),"MLP Classifier",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"MLPClassifier",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #SVM
        model = SVC(kernel='linear',random_state=0)
        model.fit(X_train,y_train)
        hasil(y_test,model.predict(X_test),"SVM",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"SVM",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #Random Forest
        model = RandomForestClassifier(n_estimators=2000,criterion='entropy',random_state=0)
        model.fit(X_train, y_train)
        hasil(y_test,model.predict(X_test),"Random Forest",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"Random Forest",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)
        hasil(y_test,model.predict(X_test),"Gaussian Naive Bayes",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"Naive Bayes",'Fold':i,'Akurasi':Acc},ignore_index=True)
        
        #KNN
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        hasil(y_test,model.predict(X_test),"KNN",i)
        Acc = accuracy_score(y_test, model.predict(X_test))
        run_kfold = run_kfold.append({'Model':"KNN",'Fold':i,'Akurasi':Acc},ignore_index=True)
        i=i+1
        
    return run_kfold

def klasifikasi(X,y):
    
    run = pd.DataFrame({},columns=['Model','Akurasi'])  
    def hasil(expected,predicted,nama):
        print("========== {} ==========".format(nama))
        print(metrics.confusion_matrix(expected,predicted))
        print(metrics.classification_report(expected,predicted))

    X_train,X_test,y_train,y_test = holdout(X,y,0.2)
    
    #Logistic Regression
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    hasil(y_test,model.predict(X_test),"Logistic Regression")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"Logistic Regression",'Akurasi':Acc},ignore_index=True)
    
    #Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    hasil(y_test,model.predict(X_test),"Decision Tree")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"Decision Tree",'Akurasi':Acc},ignore_index=True)
    
    #MLPClassifier
    model = MLPClassifier(activation='logistic', solver='lbfgs', alpha = 1e-5, 
                        hidden_layer_sizes=(16,14), random_state=0,max_iter=100000)
    model.fit(X_train, y_train)
    hasil(y_test,model.predict(X_test),"MLP Classifier")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"MLPClassifier",'Akurasi':Acc},ignore_index=True)
    
    #SVM
    model = SVC(kernel='linear',random_state=0)
    model.fit(X_train,y_train)
    hasil(y_test,model.predict(X_test),"SVM")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"SVM",'Akurasi':Acc},ignore_index=True)
    
    #Random Forest
    model = RandomForestClassifier(n_estimators=2000,criterion='entropy',random_state=0)
    model.fit(X_train, y_train)
    hasil(y_test,model.predict(X_test),"Random Forest")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"Random Forest",'Akurasi':Acc},ignore_index=True)
    
    #Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    hasil(y_test,model.predict(X_test),"Gaussian Naive Bayes")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"Naive Bayes",'Akurasi':Acc},ignore_index=True)
    
    #KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    hasil(y_test,model.predict(X_test),"KNN")
    Acc = accuracy_score(y_test, model.predict(X_test))
    run = run.append({'Model':"KNN",'Akurasi':Acc},ignore_index=True)
    
    return run
    
def feature_PCA(X):
    pca = PCA(n_components=21) #Coba-coba n_components nya
    principalComponents = pca.fit_transform(X)
    variance_ratio = pca.explained_variance_ratio_
    print(variance_ratio)
    return principalComponents


run = pd.DataFrame({},columns=['Model','Akurasi'])  
run_kfold = pd.DataFrame({},columns=['Model','Fold','Akurasi'])  
df = pd.read_csv("CTG.csv")
df = df.replace('?',np.nan)
df = prep(df)

#3 Class Classification
fitur = ['AC','FM','UC','DL','DS','DP','LB','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax',
         'Nzeros','Mode','Mean','Median','Variance','Tendency']
X = np.array(df[fitur])
y = np.array(df['NSP'])

#Preproses
X = preproses_minmax(X) #MinMax
#X = preproses_standarize(X) #StandardScaler

#X = feature_PCA(X)

#Classification
run = klasifikasi(X,y)
run_kfold = klasifikasi_kfold(X,y)

