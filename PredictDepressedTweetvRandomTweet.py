#!/usr/bin/env python
# coding: utf-8

# ## Depression in Tweets

# In[46]:


# import nltk library
import nltk; nltk.download('punkt')
from nltk import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer

# import stopword libraries
nltk.download('stopwords'); from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

# import other libraries
import pandas as pd
import numpy as np
import string
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, GridSearchCV

# import word embedding library
#import glove_helper

# import helper libraries
import collections
from common import utils, vocabulary

#display multiple results per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#export models
from sklearn.externals import joblib


# In[47]:


#read in tweets
#df = pd.DataFrame.from_csv('../depression_tweets.csv', header=None, parse_dates=True, infer_datetime_format=True)
df = pd.read_csv("depression_tweets.csv",encoding='utf-8')

# In[48]:


#set column names
#df.columns = ['date','tweet_id', 'handle', 'id', 'tweet', 'language', 'device', 'notes', 'notes_2']


# In[ ]:


#look at data
df.head(5)


# In[ ]:


#how man non-distinct tweets
len(df)


# In[ ]:


#filter to english only
df = df[df['language'] == 'en']


# In[ ]:


#how many tweets now
len(df)


# In[ ]:


#any users w/lots of tweets that might skew model?
#not any that seem too high
df['handle'].value_counts().head(5)


# In[ ]:


#how many distinct tweets
len(df.tweet.unique())


# In[ ]:


#make distinct tweets the df
df = pd.DataFrame(df.tweet.unique())


# In[ ]:


#rename columns
df.columns = ['tweets']


# In[ ]:


#export sample to check quality
# pd.options.display.max_colwidth = 1000
# df_sample = df.sample(n=100)
# df_sample.to_csv('../sample_100_depression_tweets.csv')


# In[ ]:


#look up specific tweet
pd.options.display.max_colwidth = 10000
df.iloc[45055]


# In[ ]:


#create column on 1's
x = [1]
x = x * len(df)
df['target'] = x


# In[ ]:


df.head(5)


# ## Bring in random tweets

# In[ ]:


#read in tweets
df_2 = pd.read_csv('random_sample_tweets_11k2018-04-11_21-28-23.csv',encoding='utf-8')


# In[ ]:


#look at data
df_2.head()


# In[ ]:


#how many
len(df_2)


# In[ ]:





# In[ ]:


#how many distinct tweets
len(df_2.tweets.unique())


# In[ ]:


#Make dataframe of unique
df_2 = pd.DataFrame(df_2.tweets.unique())

#give column name
df_2.columns = ['tweets']


# In[ ]:


#make all tweets lowercase
df_2['tweets'] = df_2['tweets'].str.lower()
df_2.columns = ['tweets']


# In[ ]:


df_2.head()


# In[ ]:


#check for tweets that use depression
df_2[(df_2['tweets'].str.contains('depressed') | df_2['tweets'].str.contains('depression'))]

#drop them
df_2.drop(df_2[(df_2.tweets.str.contains('depressed')) | (df_2.tweets.str.contains('depression'))].index, inplace=True)


# In[ ]:


#recheck length
len(df_2)


# In[ ]:


#export to check quality
# df_2_sample = df_2.sample(n=100)
# df_2_sample.to_csv('../sample_100_random_tweets.csv')


# In[ ]:


#column of 0's
x = 0
x = x * len(df_2)

df_2['target'] = x


# In[ ]:


#balance classes
df_3 = df.sample(n=len(df_2))


# In[ ]:


# df_3.head()


# In[ ]:


#combine dfs
df = pd.concat([df_3,df_2])


# In[ ]:


len(df)


# In[ ]:


#preprocess tweets
example_text="""'RT @techreview: A neural network can 
detect depression and mania in bipolar subjects 
by analyzing how they hold and tap on their smartphone…'"""

# tokenize
def tokenize_text(input_text):
    """
    Args: 
    input_text: a string representing an 
    individual review
        
    Returns:
    input_token: a list containing stemmed 
    tokens, with punctutations removed, for 
    an individual review
        
    """
    input_tokens=[]
        
    # Split sentence
    sents=sent_tokenize(input_text)
            
    # Split word
    for sent in sents:
        input_tokens+=TreebankWordTokenizer().tokenize(sent)
        
    return input_tokens


# canonicalize
def canonicalize_tokens(input_tokens):
    """
    Args:
    input_tokens: a list containing tokenized 
    tokens for an individual review
    
    Returns:
    input_tokens: a list containing canonicalized 
    tokens for an individual review
    
    """
    input_tokens=utils.canonicalize_words(input_tokens)
    return input_tokens


# preprocessor 
def preprocessor(raw_text):
    """
    Args:
    raw_text: a string representing an
    individual review
    
    Returns:
    preprocessed_text: a string representing 
    a preprocessed individual review
    
    """
    # tokenize
    tokens=tokenize_text(raw_text)
    
    # canonicalize
    canonical_tokens=canonicalize_tokens(tokens)
    
    # rejoin string
    preprocessed_text=(" ").join(canonical_tokens) 
    return preprocessed_text

# example data
#input_tokens=tokenize_text(example_text)
#print(input_tokens)

#canonical_tokens=canonicalize_tokens(input_tokens)
#print(canonical_tokens)

preprocessed_text=preprocessor(example_text) 
print(preprocessed_text)


# In[ ]:


# examine stopwords

# sklearn stopwords (frozenset)
sklearn_stopwords=stop_words.ENGLISH_STOP_WORDS
print("number of sklearn stopwords: %d" %(len(sklearn_stopwords)))
#print(sklearn_stopwords)

# nltk stopwords (list)
nltk_stopwords=stopwords.words("english")
print("number of nltk stopwords: %d" %(len(nltk_stopwords)))
#print(nltk_stopwords)

# combined sklearn, nltk, other stopwords (set)
total_stopwords=set(list(sklearn_stopwords.difference(set(nltk_stopwords)))+nltk_stopwords)

other_stopwords=["DG", "DGDG", "@", "rt", "'rt", "'", ":", "depression", "depressed", "RT"]
for w in other_stopwords:
    total_stopwords.add(w)
    
print("number of total stopwords: %d" %(len(total_stopwords)))


# In[ ]:


#look at review w/o stop words
new_review = []
for i in preprocessed_text.split():
    if i in total_stopwords:
        continue
    else:
        new_review.append(i)
        
print(new_review)


# In[ ]:


#reset index
df = df.reset_index(drop=True)


# In[ ]:


#split into test, train before sampling to belance
# using recoded labels
#create train, test data
df['is_train'] = np.random.uniform(0,1, len(df)) <= .8

train_data, test_data = df[df['is_train'] == True], df[df['is_train'] == False]

# examine train, test shapes
print("train, test set size: %d, %d" %(len(train_data), len(test_data))) # train_data: 129023, test_data: 32256
print("")

# examine train set examples
print("example:")
print("tweet: %s" %(train_data.get_value(5,'tweets')))
print("label: %s" %(train_data.get_value(5,'target')))


# In[ ]:


#check class balance
train_data['target'].value_counts()

# In[ ]:

print("example:")
print("tweet: %s" %(train_data.get_value(32,'tweets')))
print("label: %s" %(train_data.get_value(32,'target')))

# ## Logistic Regression

# In[ ]:


#build tf-idf model
vec=TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1,3), stop_words=total_stopwords, max_features=10000)
vec_train_data=vec.fit_transform(train_data['tweets']) 
vec_test_data=vec.transform(test_data['tweets']) 


# In[ ]:


# train Logistic Regression
logit=LogisticRegression(penalty='l2')
logit.fit(vec_train_data, train_data['target'])
pred_labels=logit.predict(vec_test_data)
    
# assess model
f1=f1_score(test_data['target'], pred_labels, average="weighted") 
accuracy=accuracy_score(test_data['target'], pred_labels)
confusion=confusion_matrix(test_data['target'], pred_labels)
print("logistic regression f1 score: %.3f" %(f1))
print("logistic regression accuracy score: %.3f" %(accuracy))
print("logistic regression confusion matrix:")
print(confusion)


# In[49]:


#try Keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[ ]:


#create integer encoding of docs
vocab_size = 100
encoded_docs = [one_hot(d, vocab_size) for d in df['tweets']]


# In[ ]:


#try tokenizer instead
t = Tokenizer()
t.fit_on_texts(df['tweets'])
vocab_t_size = len(t.word_index) + 1


# In[ ]:


#create sequence
encoded_t_docs = t.texts_to_sequences(df['tweets'])


# In[ ]:


# pad docs to equals size
pad = 40
# padded_docs = pad_sequences(encoded_docs, maxlen=pad, padding='post')
padded_t_docs = pad_sequences(encoded_t_docs, maxlen=pad, padding='post')


# In[ ]:


#padded_docs[11105]


# In[ ]:


from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(padded_docs, df['target'], test_size=.8)
X_train,X_test,Y_train,Y_test = train_test_split(padded_t_docs, df['target'], test_size=.8)


# In[ ]:


X_train.shape


# In[ ]:


# create the model
embedding_size = 32

model = Sequential()
# model.add(Embedding(vocab_size, embedding_size, input_length=pad))
model.add(Embedding(vocab_t_size, embedding_size, input_length=pad))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


# Fit the model
epochs=3
batch_size=128

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


keras_journal = ["Sometime I feel very alone and anxious"]


# In[ ]:


encoded_journal = t.texts_to_sequences(keras_journal)


# In[ ]:


encoded_journal


# In[ ]:


#pad
pad = 40
padded_journal = pad_sequences(encoded_journal, maxlen=pad, padding='post')


# In[ ]:


padded_journal


# In[ ]:


ynew = model.predict_proba(padded_journal)


# In[ ]:


ynew


# In[ ]:


#get top words
#look at top 5 weights for each class
#get coefficients for all features
coef_sq = logit.coef_

#get index of top 5 absolute values for each class
weight_indx = np.argsort(coef_sq)[:, -20:]

#flatten so can use to look up wieghts
weight_indx = weight_indx.flatten()

#get coefficients based on index
weights = coef_sq[:, weight_indx]
 
#get words that match weights based on index
vocab = np.array(vec.get_feature_names())[weight_indx]

# make table
df = pd.DataFrame({'Weights of words that predict depression': weights[0]}
                  , index=vocab)
df


# In[ ]:


#try to make up an example journal
journal = """I'm not the only traveller. Who has not repaid his debt. I've been searching for a trail to follow, again. Take me back to the night we met."""

#score test journal
vec_test_example=vec.transform([journal]) 
print("probability of class 0 and 1: ",logit.predict_proba(vec_test_example))

#get words and weights from test journal
word_idx = np.nonzero(vec_test_example)[1]
vocab = np.array(vec.get_feature_names())[word_idx]
weights = coef_sq[:, word_idx]
df = pd.DataFrame({'Weights of words in sample Journal': weights[0]}
                  , index=vocab)
df.sort_values(by='Weights of words in sample Journal')


# In[ ]:


#export tfidf model
tfidf_file = 'tfidf_exported_model'
joblib.dump(vec, tfidf_file)


# In[ ]:


#export logistic regression
logistic_regression_file = 'logistic_regression_model'
joblib.dump(logit, logistic_regression_file)


# In[ ]:


#test out exported models against prev sample journal
loaded_tfidf = joblib.load('tfidf_exported_model')
loaded_lr = joblib.load('logistic_regression_model')

#score test journal
export_test_example=loaded_tfidf.transform([journal]) 
print("probability of class 0 and 1: ",loaded_lr.predict_proba(export_test_example))