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

import os
import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import extract
import categorize

from werkzeug.utils import secure_filename

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

import risk_ews as dews


app = Flask(__name__)
app.debug = True

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



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

def fuzzy(a,b,c):
    a1=''
    a2=''
    a3=''
    if(a>0 and a<0.34):
        a1='low'
    elif(a>0.34 and a<0.68):
        a1='medium'
    else:
        a1='high'
    if(b>0 and b<0.34):
        b1='low'
    elif(b>0.34 and b<0.68):
        b1='medium'
    else:
        b1='high'
    if(c==0):
        c1='low'
    else:
        c1='high'
    if(a1=='low' and b1=='low' and c1=='low'):
        return ('very low')
    if(a1=='low' and b1=='low' and c1=='high'):
        return ('low')
    if(a1=='low' and b1=='medium' and c1=='low'):
        return('low')
    if(a1=='low' and b1=='high' and c1=='low'):
        return('low')
    if(a1=='low' and b1=='medium' and c1=='high'):
        return('low')
    if(a1=='low' and b1=='high' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='low' and c1=='low'):
        return('low')
    if(a1=='medium' and b1=='low' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='medium' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='medium' and c1=='low'):
        return('medium')
    if(a1=='medium' and b1=='high' and c1=='low'):
        return('medium')
    if(a1=='medium' and b1=='high' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='low' and c1=='low'):
        return('medium')
    if(a1=='high' and b1=='low' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='medium' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='medium' and c1=='low'):
        return('high')
    if(a1=='high' and b1=='high' and c1=='low'):
        return('high')
    if(a1=='high' and b1=='high' and c1=='high'):
        return('very high')


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

X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []
y_t1 = []
y_t2 = []
y_t3 = []
y_t4 = []
y_t5 = []
y_t6 = []
y_t7 = []
y_t8 = []
page_ids = []




if os.path.isfile("label_list"):
    #print ("Info: label_list found.")
    #=================================================================
    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()
            
            baseline_angle = float(content[0])
            X_baseline_angle.append(baseline_angle)
            
            top_margin = float(content[1])
            X_top_margin.append(top_margin)
            
            letter_size = float(content[2])
            X_letter_size.append(letter_size)
            
            line_spacing = float(content[3])
            X_line_spacing.append(line_spacing)
            
            word_spacing = float(content[4])
            X_word_spacing.append(word_spacing)
            
            pen_pressure = float(content[5])
            X_pen_pressure.append(pen_pressure)
            
            slant_angle = float(content[6])
            X_slant_angle.append(slant_angle)
            
            trait_1 = float(content[7])
            y_t1.append(trait_1)
            
            trait_2 = float(content[8])
            y_t2.append(trait_2)
            
            trait_3 = float(content[9])
            y_t3.append(trait_3)
            
            trait_4 = float(content[10])
            y_t4.append(trait_4)

            trait_5 = float(content[11])
            y_t5.append(trait_5)

            trait_6 = float(content[12])
            y_t6.append(trait_6)

            trait_7 = float(content[13])
            y_t7.append(trait_7)
            
            trait_8 = float(content[14])
            y_t8.append(trait_8)
            
            page_id = content[15]
            page_ids.append(page_id)
    #===============================================================
    
    # emotional stability
    X_t1 = []
    for a, b in zip(X_baseline_angle, X_slant_angle):
        X_t1.append([a, b])
    
    # mental energy or will power
    X_t2 = []
    for a, b in zip(X_letter_size, X_pen_pressure):
        X_t2.append([a, b])
        
    # modesty
    X_t3 = []
    for a, b in zip(X_letter_size, X_top_margin):
        X_t3.append([a, b])
        
    # personal harmony and flexibility
    X_t4 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t4.append([a, b])
        
    # lack of discipline
    X_t5 = []
    for a, b in zip(X_slant_angle, X_top_margin):
        X_t5.append([a, b])
        
    # poor concentration
    X_t6 = []
    for a, b in zip(X_letter_size, X_line_spacing):
        X_t6.append([a, b])
        
    # non communicativeness
    X_t7 = []
    for a, b in zip(X_letter_size, X_word_spacing):
        X_t7.append([a, b])
    
    # social isolation
    X_t8 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t8.append([a, b])
    
    #print X_t1
    #print type(X_t1)
    #print len(X_t1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_t1, y_t1, test_size = .30, random_state=8)
    clf1 = SVC(kernel='rbf')
    clf1.fit(X_train, y_train)
    #print ("Classifier 1 accuracy: ",accuracy_score(clf1.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t2, y_t2, test_size = .30, random_state=16)
    clf2 = SVC(kernel='rbf')
    clf2.fit(X_train, y_train)
    #print ("Classifier 2 accuracy: ",accuracy_score(clf2.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t3, y_t3, test_size = .30, random_state=32)
    clf3 = SVC(kernel='rbf')
    clf3.fit(X_train, y_train)
    #print ("Classifier 3 accuracy: ",accuracy_score(clf3.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t4, y_t4, test_size = .30, random_state=64)
    clf4 = SVC(kernel='rbf')
    clf4.fit(X_train, y_train)
    #print ("Classifier 4 accuracy: ",accuracy_score(clf4.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t5, y_t5, test_size = .30, random_state=42)
    clf5 = SVC(kernel='rbf')
    clf5.fit(X_train, y_train)
    #print ("Classifier 5 accuracy: ",accuracy_score(clf5.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t6, y_t6, test_size = .30, random_state=52)
    clf6 = SVC(kernel='rbf')
    clf6.fit(X_train, y_train)
    #print ("Classifier 6 accuracy: ",accuracy_score(clf6.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t7, y_t7, test_size = .30, random_state=21)
    clf7 = SVC(kernel='rbf')
    clf7.fit(X_train, y_train)
    #print ("Classifier 7 accuracy: ",accuracy_score(clf7.predict(X_test), y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t8, y_t8, test_size = .30, random_state=73)
    clf8 = SVC(kernel='rbf')
    clf8.fit(X_train, y_train)
    
def scoring(arr):
    sum=0
    max1=arr[0]
    max2=arr[0]
    for i in range(8):
        arr[i]=float(arr[i])
    for i in range(4):
        arr[i]=1-arr[i]
    for i in range(8):
        sum+=arr[i]
    return sum/8



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def res():
    return render_template('result.html')
   
@app.route('/learning', methods=['POST','GET'])
def learning():
    score1=0
    score2=0
    score3=0
    score4=0
    score5=0
    score6=0
    score7=0
    score8=0

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

        file1 = request.files['file1']
        file2 = request.files['file2']
        file3 = request.files['file3']
        file4 = request.files['file4']
        file5 = request.files['file5']

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename) and file3 and allowed_file(file3.filename) and file4 and allowed_file(file4.filename) and file5 and allowed_file(file5.filename):
            file_names=[secure_filename(file1.filename),secure_filename(file2.filename),secure_filename(file3.filename),secure_filename(file4.filename),secure_filename(file5.filename)]

            for file_name in file_names:
                    
                raw_features = extract.start(file_name)
                
                raw_baseline_angle = raw_features[0]
                baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
                #print ("Baseline Angle: "+comment)
                
                raw_top_margin = raw_features[1]
                top_margin, comment = categorize.determine_top_margin(raw_top_margin)
                #print ("Top Margin: "+comment)
                
                raw_letter_size = raw_features[2]
                letter_size, comment = categorize.determine_letter_size(raw_letter_size)
                #print ("Letter Size: "+comment)
                
                raw_line_spacing = raw_features[3]
                line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
                #print ("Line Spacing: "+comment)
                
                raw_word_spacing = raw_features[4]
                word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
                #print ("Word Spacing: "+comment)
                
                raw_pen_pressure = raw_features[5]
                pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
                #print ("Pen Pressure: "+comment)
                
                raw_slant_angle = raw_features[6]
                slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
                #print ("Slant: "+comment)
                
                #print
                score1+=0.2*clf1.predict([[baseline_angle, slant_angle]])[0]
                score2+=0.2*clf2.predict([[letter_size, pen_pressure]])[0]
                score3+=0.2*clf3.predict([[letter_size, top_margin]])[0]
                score4+=0.2*clf4.predict([[line_spacing, word_spacing]])[0]
                score5+=0.2*clf5.predict([[slant_angle, top_margin]])[0]
                score6+=0.2*clf6.predict([[letter_size, line_spacing]])[0]
                score7+=0.2*clf7.predict([[letter_size, word_spacing]])[0]
                score8+=0.2*clf8.predict([[line_spacing, word_spacing]])[0]
                score_hand=[str(round(score1,2)),str(round(score2,2)),str(round(score3,2)),str(round(score4,2)),str(round(score5,2)),str(round(score6,2)),str(round(score7,2)),str(round(score8,2))]
        max1=0
        max2=0
        i1=0
        i2=0
        print(score_hand)
        for i in range(8): 
            if float(score_hand[i]) > max1: 
                max1 = float(score_hand[i]) 
                i1=i    
            if float(score_hand[i]) > max2 and float(score_hand[i]) <= max1 and i!=i1:
                max2 = float(score_hand[i])
                i2=i
        array=['emotional stability','mental energy or will power','modesty','personal harmony and flexibility','lack of discipline','poor concentration','non communicativeness','social isolation']
        final_score=[fuzzy(scoring(score_hand),score,y),array[i1],max1,array[i2],max2]
        return render_template('result.html', result=final_score)



if __name__ == '__main__':
    app.run()