An "Early Warning System" Supervised Learning Trigger for Student Risk

>NOTE: Sample data is included in this repository as `students.csv`

This repository contains a module of proprietary functions as `risk_ews.py` and `risk_ews_process.py` that take the user through the implementation of an "Early Warning System" (EWS) trigger for intervention in potential student risk.  The optimized trigger uses a 'feature weighted nearest neighbor' supervised learning model.  A sample dataset is included as `students.csv` in the same directory.


`risk_ews_process.py` demonstrates a method for obtaining a ["feature weighted nearest neighbor" model] using an optimized Decision Tree and a KNN model.  The final model is fit to the dataset and performance metrics accompany:

## Goals

This program file implements the supervised learning model on the dataset with the help of `risk_ews.py` that trains the model and finally returns the binary output '1' if student is at risk and '0' if student is not at risk  

## Software and Library Requirements

* Python 2.7.11
* Jupyter Notebook 4.2.2
* Numpy 1.11.2
* scikit-image 0.12.3
* matplotlib 1.5.2

## Data
-----------
The sample data used for this notebook comes from the [UCI Irvine Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) and has been modified to make it relevant to the college lifestyle and to fit a classification problem.

> "This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and college related features) and it was collected by using school reports and questionnaires. In [Cortez and Silva, 2008], the datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details)."

The sample dataset used in this project is included as `students.csv`. The last column 'passed' is the target/label, all other are feature columns.  The CSV contains a header with the following 30 attributes:

- __sex__ : student's sex (binary: "F" - female or "M" - male)
- __age__ : student's age (numeric: from 15 to 22)
- __Pstatus__ : parent's cohabitation status (binary: "T" - living together or "A" - apart)
- __Mjob__ : mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- __Fjob__ : father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- __reason__ : reason to choose this college (nominal: close to "home", college "reputation", "course" preference or "other")
- __studytime__ : weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- __failures__ : number of past semester failures (numeric: n where, 0<=n<4)
- __famsup__ : family educational support (binary: yes or no)
- __activities__ : extra-curricular activities (binary: yes or no)
- __internet__ : Internet access at room (binary: yes or no)
- __romantic__ : in a romantic relationship (binary: yes or no)
- __famrel__ : quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- __freetime__ : free time after school (numeric: from 1 - very low to 5 - very high)
- __goout__ : going out with friends (numeric: from 1 - very low to 5 - very high)
- __Dalc__ : workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- __Walc__ : weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- __health__ : current health status (numeric: from 1 - very bad to 5 - very good)
- __absences__ : number of absences in college (numeric: from 0 to 93) [this can be changed according to the college]
- __CGPA__ : current CGPA (numeric: real number from 0 to 10.0 rounded off upto 2 decimal places) 

Each student has a target that takes two discrete labels:

- __risk__ : whether a student will take extreme step (binary: 1 or 0)