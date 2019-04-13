Assessment of a particular individual's social media handle to determine mental stability

Libraries used - 
pandas
numpy
scikit-learn
nltk

Pre-trained models used - 
TF-IDF Classifier (tfidf.pkl)
Bag-of-Words Classifier (bow.pkl)

1. Preprocessed sentences using Stemmer, Lemmatizer, and removal of stop-words.
2. Training the model on TFIDF and BOW, and dumping the trained models to pickled files.
3. Classifying input sentences according to the model which obtained a better F1-score, to gain greater accuracy of prediction.
4. Giving equal weights to all scored sentences, and compiled a final fuzzified score.


