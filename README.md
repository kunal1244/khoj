THIS WAS THE TERM PROJECT FOR THE COURSE SOFT COMPUTING TOOLS IN ENGINEERING at IIT Kharagpur

A WEB Interface for an overall analysis of the Mental Health of a student visiting the Counselling Center with the help of three things:
1. HANDWRITING ANALYSIS
2. ACADEMIC, EXTRA-CURRICULAR and Personal details
3. Analysis of POSTS, COMMENTS on Social Media


A final fuzzy judgement on the risk of a depressed student going for an extreme step and his vulnerable personality traits that need to be looked after is displayed on the screen.

(I) HANDWRITING ANALYSIS by GRAPHOLOGY:  

	Graphology is defined as the analysis of the physical characteristics and patterns of the handwriting of an individual to understand his or her psychological state at the time of writing.

	The proposed methodology extracts seven handwriting features from the images of writings, namely:
		1. top margin, 
		2. pen pressure,
		3. baseline angle,
		4. letter size,
		5. line spacing,
		6. word spacing and
		7. slant angle.

	From graphology we used these rules to train the 8 SVM Classifiers for personality traits-

		1. Emotional stability trained for its dependent baseline_angle and slant_angle

		2. Will power trained for its dependent letter_size and pen_pressure

		3. Modesty trained for its dependent letter_size and top_margin

		4. Personal harmony trained for its dependent line_spacing and word_spacing

		5. Discipline trained for its dependent slant_angle, top_margin

		6. Concentration trained for its dependent letter_size and line_spacing

		7. Communicativeness trained for its dependent letter_size and word_spacing

		8. Social Isolation trained for its dependent line_spacing and word_spacing

	The test image is passed as input to the trained model via `train_predict.py` which trains eight SVM classifiers, extracts the features of the image via `extract.py` and then classifies it for every personality trait.

(II) ANALYSIS of ACADEMIC and Other details:

	Detailed description  about this is given in the README.md file in academic directory

(III) Sentiment Anaysis of Social Media tweets, posts, comments:

	#TO BE UPDATED as very crude method used

	Assessment of a particular individual's social media handle to determine mental stability
	Pre-trained models used - 
	TF-IDF Classifier (tfidf.pkl)
	Bag-of-Words Classifier (bow.pkl)

	1. Preprocessed sentences using Stemmer, Lemmatizer, and removal of stop-words.
	2. Training the model on TFIDF and BOW, and dumping the trained models to pickled files.
	3. Classifying input sentences according to the model which obtained a better F1-score, to gain greater accuracy of prediction.
	4. Giving equal weights to all scored sentences, and compiled a final fuzzified score.

# GETTING SET AND RUNNING

1. Clone this repository-
2. Everything has been implemented in Python3. Install all the requirements in requirements.txt file
3. RUN the start.py file in python3
4. Open the browser and go to http://127.0.0.1:5000/ local host
5. Enter all the details and get the result!