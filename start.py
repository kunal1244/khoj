from flask import Flask
from flask import render_template, abort, jsonify, request,redirect, json

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import pandas as pd
import numpy as np
import re
import pickle

import time

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

import risk_ews as dews


app = Flask(__name__)
app.debug = True

class TweetClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                len(list(self.tf_depressive.keys())))
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                len(list(self.tf_positive.keys())))
        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets 


    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.depressive_tweets + self.positive_tweets
        self.depressive_words = 0
        self.positive_words = 0
        self.tf_depressive = dict()
        self.tf_positive = dict()
        self.idf_depressive = dict()
        self.idf_positive = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets.iloc[i])
            count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
            for word in message_processed:
                if self.labels.iloc[i]:
                    self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                    self.depressive_words += 1
                else:
                    self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                    self.positive_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels.iloc[i]:
                    self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                else:
                    self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        self.sum_tf_idf_depressive = 0
        self.sum_tf_idf_positive = 0
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                          / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
            self.sum_tf_idf_depressive += self.prob_depressive[word]
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
            
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                          / (self.idf_depressive.get(word, 0) + self.idf_positive[word]))
            self.sum_tf_idf_positive += self.prob_positive[word]
        for word in self.tf_positive:
            self.prob_positive[word] = (self.prob_positive[word] + 1) / (self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
            
    
        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets 
                    
    def classify(self, processed_message):
        pDepressive, pPositive = 0, 0
        for word in processed_message:                
            if word in self.prob_depressive:
                pDepressive += log(self.prob_depressive[word])
            else:
                if self.method == 'tf-idf':
                    pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                else:
                    pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
            if word in self.prob_positive:
                pPositive += log(self.prob_positive[word])
            else:
                if self.method == 'tf-idf':
                    pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys()))) 
                else:
                    pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
            pDepressive += log(self.prob_depressive_tweet)
            pPositive += log(self.prob_positive_tweet)
        return pDepressive >= pPositive



def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

classifier_f = open("tfidf.pkl", "rb")
sc_tf_idf = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("bow.pkl", "rb")
sc_bow = pickle.load(classifier_f)
classifier_f.close()

#ACADEMIC DATA




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST','GET'])
def learning():
	if request.method=='POST':
		data1=request.form['userinput1']
		data2=request.form['userinput2']
		data3=request.form['userinput3']
		data4=request.form['userinput4']
		data5=request.form['userinput5']
		response1=int(sc_tf_idf.classify(process_message(data1)))
		response2=int(sc_tf_idf.classify(process_message(data2)))
		response3=int(sc_tf_idf.classify(process_message(data3)))
		response4=int(sc_tf_idf.classify(process_message(data4)))
		response5=int(sc_tf_idf.classify(process_message(data5)))
		score=0.2*(response1+response2+response3+response4+response5)
		return render_template('result.html',result=score)

@app.route('/acads', methods=['POST','GET'])
def acads():
	if request.method=='POST':
		sex=request.form['sex']
		age=request.form['age']
		p_status=request.form['p_status']
		mjob=request.form['mjob']
		fjob=request.form['fjob']
		reason=request.form['reason']
		stime=request.form['stime']
		fail=request.form['fail']
		famsup=request.form['famsup']
		act=request.form['act']
		net=request.form['net']
		rel=request.form['rel']
		famrel=request.form['famrel']
		free=request.form['free']
		goout=request.form['goout']
		dalc=request.form['dalc']
		walc=request.form['walc']
		health=request.form['health']
		abscences=request.form['abscences']
		cgpa=request.form['cgpa']
		data=[sex,age,p_status,mjob,fjob,reason,stime,fail,famsup,act,net,rel,famrel,free,goout,dalc,walc,health,abscences]
		with open('testing.csv','a') as fd:
			for i in data:
				fd.write(i)
				fd.write(',')
			fd.write(cgpa)
			fd.write('\n')
		fd.close()
		
		student_data = pd.read_csv("students.csv", encoding='utf-8')
		test = pd.read_csv("testing.csv", encoding='utf-8')
		n_students = student_data.shape[0]
		n_features = student_data.shape[1] - 1
		n_passed = student_data["risk"].value_counts()[1]
		n_failed = student_data["risk"].value_counts()[0]
		grad_rate = float(n_passed)/n_students*100


		feature_cols = list(student_data.columns[:-1])  # all columns but last are features
		target_col = student_data.columns[-1]  # last column is the target/label

		X_all = student_data[feature_cols]  # feature values for all students
		y_all = student_data[target_col]  # corresponding targets/labels

		test_example = test[feature_cols]

		X_all = dews.preprocess_features(X_all)
		test_example = dews.preprocess_features(test_example)
		test_example = test_example.iloc[-1:]

		num_all = student_data.shape[0]  # same as len(student_data)
		num_train = 316  # about 80% of the data
		num_test = 79

		X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,train_size=num_train,test_size=num_test,stratify=y_all)


		f1_scorer = make_scorer(f1_score, pos_label=1)

		parameters = {'max_depth': range(1,15)}
		dt = DecisionTreeClassifier()
		grid_search = GridSearchCV(dt,parameters,scoring=f1_scorer)
		grid_search.fit(X_train,y_train)

		dt_tuned = DecisionTreeClassifier(max_depth=3)
		dt_tuned.fit(X_train,y_train)

		# Subset the Dataset by removing features whose 'importance' is zero, 
		# according to a tuned Decision tree in 1.1 
		sub = np.nonzero(dt_tuned.feature_importances_)[0].tolist()
		subset_cols = list(X_train.columns[sub])
		X_train_subset = X_train[subset_cols]
		X_test_subset = X_test[subset_cols]
		test_example_subset = test_example[subset_cols]

		clf_default = KNeighborsClassifier()

		# Determine the number of nearest neighbors that optimizes accuracy 
		parameters = {'n_neighbors': range(1,30)}
		knn = KNeighborsClassifier()
		knn_tuned = GridSearchCV(knn,parameters,scoring=f1_scorer)
		knn_tuned.fit(X_train_subset,y_train)
		clf_tuned = KNeighborsClassifier(n_neighbors=knn_tuned.best_params_['n_neighbors'])
		y = dews.train_predict("Subset_KNN", X_train_subset, y_train, X_test_subset, y_test, test_example_subset, 300, clf_default, clf_tuned)


		return render_template('result1.html',result=y)

if __name__ == '__main__':
    app.run()