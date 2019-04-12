

import numpy as np
import pandas as pd #Successfully installed pandas-0.19.2
import time

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

import risk_ews as dews


student_data = pd.read_csv("students.csv", encoding='utf-8')
test = pd.read_csv("testing.csv", encoding='utf-8')
#print ("Student data read successfully!")


# EXPLORE THE DATA
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
#print "Feature values:"
#print X_all.head()  # print the first 5 rows
# CREATE DUMMY BINARY VARS FOR ALL CATEGORICAL FEATURES
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

print ("The predicted result, whether a student will take an extreme step is: "+str(y))