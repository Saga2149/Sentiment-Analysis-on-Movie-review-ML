
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import itertools
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier


# In[3]:


class SentimentAnalysis(object):
    def __init__(self, trainpkl, testpkl, train, test):
        self.X_train=trainpkl
        self.X_test_submit=testpkl
        self.train=train
        self.test=test
    def logistic(self):
        print("Logistic Regression")
        LogRegGrid_path = 'save/LogRegGrid.pkl'
        GridSe1 = pickle.load(open(LogRegGrid_path,"rb"))
        print("tuned hpyerparameters :(best parameters) ",GridSe1.best_params_)
        print("accuracy :",GridSe1.best_score_)
        pred1=GridSe1.predict(self.X_test_submit)
        first_sub=test.copy()
        n=first_sub.shape[0]
        first_sub["Sentiment"] = pred1
        first_sub = first_sub.loc[:, ["PhraseId", "Sentiment"]]
        first_sub.to_csv("logistic.csv", index=False)
        

    def xgboost(self):
        print("XG Boost")
        XgBoost_path = 'save/XgBoost.pkl'
        GridSe2 = pickle.load(open(XgBoost_path,"rb"))
        print("tuned hpyerparameters :(best parameters) ",GridSe2.best_params_)
        print("accuracy :",GridSe2.best_score_)
        pred2=GridSe2.predict(self.X_test_submit)
        first_sub=self.test.copy()
        n=first_sub.shape[0]
        first_sub["Sentiment"] = pred2
        first_sub = first_sub.loc[:, ["PhraseId", "Sentiment"]]
        first_sub.to_csv("XgBoost.csv", index=False)
        
    def naivebayes(self):
        print("Naive Bayes")
        MultinomialNB_path = 'save/MultinomialNB.pkl'
        GridSe3 = pickle.load(open(MultinomialNB_path,"rb"))
        print("tuned hpyerparameters :(best parameters) ",GridSe3.best_params_)
        print("accuracy :",GridSe3.best_score_)
        pred3=GridSe3.predict(self.X_test_submit)
        first_sub=self.test.copy()
        n=first_sub.shape[0]
        first_sub["Sentiment"] = pred3
        first_sub = first_sub.loc[:, ["PhraseId", "Sentiment"]]
        first_sub.to_csv("MultinomialNB.csv", index=False)
        
    def ensemble(self):
        model = 'save/Ensemble.pkl'
        GridSe4 = pickle.load(open(model,"rb"))
        pred4=GridSe4.predict(self.X_test_submit)
        first_sub=self.test.copy()
        n=first_sub.shape[0]
        first_sub["Sentiment"] = pred4
        first_sub = first_sub.loc[:, ["PhraseId", "Sentiment"]]
        first_sub.to_csv("Ensembled.csv", index=False)
    
    def visualize(self):
        dist = self.train.groupby(["Sentiment"]).size()
        dist = dist / dist.sum()
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(dist.keys(), dist.values)
        plt.show()

if __name__ == "__main__":
    X_train = pickle.load(open("save/X_train.pkl","rb"))
    X_test_submit = pickle.load(open("save/X_test_submit.pkl","rb"))
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    load=SentimentAnalysis(X_train,X_test_submit,train,test)
    load.visualize()
    load.logistic()
    load.xgboost()
    load.naivebayes()
    load.ensemble()
        

