# First we will set up our variahbles declaration and imports

# coding: utf-8
# Dataset of annoted tweet can be found : http://help.sentiment140.com/for-students/
# 
# 0 -> negative
# 2 -> neutral
# 4 -> positive

# In[1]:

# please make sure that you have downloaded -  nltk.download('vader_lexicon')

import csv
import json
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as sw
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from random import shuffle
import random
import numpy 
from sklearn.model_selection import cross_val_score


#        to do: cross validation, solve the no-netural problem with sentiment140 training data (we have done this by creating multiple other datasets),, Naive Bayes models
# Current best model is Model1_3 One vs rest LinearSVM with tdif vectorizer, with current parameters F1_score [ 0.80373832  0.69135802  0.73214286]

#Declare X and Y variables, percentage split, hyperparameters and constraints 

X1 = [] #data
Y1 = [] #label
X2 = [] #data
Y2 = [] #label
X3 = [] #data
Y3 = [] #label
X4 = [] #data
Y4 = [] #label
Y4_vader = [] #label

percentagesplit=0.3
SVM= LinearSVC() #define type of SVM that will be used

sentdict=0 #restrict features to a particular feature list: 0=No, 1=Vader
ngram_min=1 #for anything greater than 1 sentdict must equal 0
ngram_max=3
 
#Variables for data distribution calculations
num_4 = 0
num_2 = 0
num_0 = 0

#Datasets
dataset1='testdata.manual.2009.06.14.csv'
dataset2='training.1600000.processed.noemoticon.csv'
dataset3='VadaDatasetTweets.txt'
dataset4='TweetsAirline.csv' #airline tweets
#dataset5=GOPtwitterdatabase an option if we want it


# Next we load in our training and testing data, exploring various datasets for train/testing to select the best model parameters.
# We will classify 0 as negative, 2 as neutral and 4 as positve.
# In[2]:

#loading and Saving tweets from databases:

#Dataset1 - 500 tweets 182 pos, 139 neut, 177 neg
file = open(dataset1)
lines = csv.reader(file)

for line in lines :
    X1.append(line[5])
    Y1.append(int(line[0]))
    
#Dataset2 - 1600 tweets 800,000 pos and 800,00 negative since it is so big we take a subset of everything to not manage too much data
file = open(dataset2)
#lines = csv.reader(file)
lines = file.readlines()
shuffle(lines)

for line in lines[:2000]:
    line = line.split(',')
    X2.append(line[5])
    Y2.append(int(line[0][1]))
       
#Dataset3 - VADER dataset 
file = open(dataset3)

line = file.readline()

while line :
    
    line = line.split('\t')
    
    X3.append(line[2])
    
#    Thresholds selected upon analysis of the tweets and splits 2042 positve, 1227 neutral, 931 negative
    
    if float((line[1]))>=0.8: #positive sentiment
        Y3.append(4)
    elif float(line[1])<=-0.8: #negativesentiment
        Y3.append(0)
    else: 
        Y3.append(2)
    
    
    line = file.readline()
    
#Dataset4 - 14641 tweets 2363 pos, 3100 neut, 9178 neg
file = open(dataset4,encoding="utf8")
lines = csv.reader(file)

for line in lines:
    X4.append(line[10])
    Y4_vader.append(line[2])
    
    if line[1]=='positive': #positive sentiment
        Y4.append(4)
    elif line[1]=='negative': #negativesentiment
        Y4.append(0)
    else: 
        Y4.append(2)
   #as data is skewed lets extract 3000 negatives
idx_of_rand_negatives =  random.sample( [i for i, x in enumerate(Y4) if x == 0],3000)
idx_of_pos =  random.sample( [i for i, x in enumerate(Y4) if x == 4],2363)
idx_of_neutr=  random.sample( [i for i, x in enumerate(Y4) if x == 2],3000)
idx_2_keep = idx_of_rand_negatives+idx_of_pos+idx_of_neutr
X4= [X4[i] for i in idx_2_keep]
Y4 = [Y4[i] for i in idx_2_keep]
    
    
## here we select how we construct our training/test data base by adding the previous datasets    
#X = X1 + X2 + X3 + X4
#Y = Y1 + Y2 + Y3 + Y4

##Or choose only 1 particular dataset
X = X1
Y = Y1  
    
# In[5]:
#Generate X_train, X_test by splitting X
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=percentagesplit,random_state=12)
#Another way X_train=X1+X2, X_test=X3+X4

#Counting data distribution
for line in X_train:
    if line == 4:
        num_4 += 1
    elif line== 2:
        num_2 += 1
    elif line == 0:
        num_0 += 1

# In[6]:
        

if sentdict==0: #Do not restrict the vectorizer to specific features
    sentiment_dict=None
elif sentdict==1: #Use the vader dataset to extract significant feature words from the data
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict=SentimentIntensityAnalyzer.make_lex_dict(analyzer).keys()

# We are going to use the scikit-learn tf-idf to tranfrom the data https://en.wikipedia.org/wiki/Tf%E2%80%93idf and http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting than might be better that the bag of word.

# In[7]:
#####===================================
##   Different vectorizers (count, binary, tfidf)
###================================
#Create different vectorizers 


vectorizer1 = CountVectorizer(dtype=int,ngram_range=(ngram_min, ngram_max),lowercase=True,max_df=0.8, min_df=1,stop_words='english', vocabulary= sentiment_dict, binary=True)

vectorizer2=CountVectorizer(dtype=int,ngram_range=(ngram_min, ngram_max),lowercase=True,max_df=0.8, min_df=1,stop_words='english', vocabulary= sentiment_dict) #

vectorizer3=TfidfVectorizer(ngram_range=(ngram_min, ngram_max), min_df=1, #Suppress word that appear in less than 10 docs
                             max_df = 0.8, #Suppress word that appear in more than 80% of the doc
                             sublinear_tf=True,
                             smooth_idf = True, # adds “1” to the numerator and denominator as if an extra document was seen containing every term in the collection exactly once, which prevents zero divisions:

                             use_idf=True, dtype=int,stop_words='english', vocabulary= sentiment_dict ) #
#Transform data
#Binary
train_vectors1 = vectorizer1.fit_transform(X_train) 
test_vectors1 = vectorizer1.transform(X_test)

#Count
train_vectors2 = vectorizer2.fit_transform(X_train) 
test_vectors2 = vectorizer2.transform(X_test)

#TfIdf
train_vectors3= vectorizer3.fit_transform(X_train) 
test_vectors3 = vectorizer3.transform(X_test)

# Now we have to look at a given classifier.

## In[8]:
#
##from sklearn.tree import DecisionTreeClassifier
##clf = DecisionTreeClassifier() #We can put whatever we can (SVM, Naive bayes...)
##clf.fit(train_vectors,Y_train)
#
#
## =============================================================================
## #SVM Model 1 and 2 with different vectorizers
## =============================================================================

print('Training SVM Classifiers.....')
#models _1: binary vectorizers, _2: Count _3:Tfidf

model1_1 = OneVsRestClassifier(SVM) 
model1_2 = OneVsRestClassifier(SVM)
model1_3 = OneVsRestClassifier(SVM)
model2_1=OneVsOneClassifier(SVM)
model2_2=OneVsOneClassifier(SVM)
model2_3=OneVsOneClassifier(SVM)

model1_1.fit(train_vectors1,Y_train)
model2_1.fit(train_vectors1,Y_train)

model1_2.fit(train_vectors2,Y_train)
model2_2.fit(train_vectors2,Y_train)

model1_3.fit(train_vectors3,Y_train)
model2_3.fit(train_vectors3,Y_train)

##Run cross-validation to test split for each model vectorizer pair
cv_scores1_1 = cross_val_score(model1_1, train_vectors1, Y_train, cv=5,scoring='f1_macro')
cv_scores1_2= cross_val_score(model1_2, train_vectors2, Y_train, cv=5,scoring='f1_macro')
cv_scores1_3 = cross_val_score(model1_3, train_vectors3, Y_train, cv=5,scoring='f1_macro')
cv_scores2_1 = cross_val_score(model2_1, train_vectors1, Y_train, cv=5,scoring='f1_macro')
cv_scores2_2= cross_val_score(model2_2, train_vectors2, Y_train, cv=5,scoring='f1_macro')
cv_scores2_3 = cross_val_score(model2_3, train_vectors3, Y_train, cv=5,scoring='f1_macro')

print('cv_scores1_1', cv_scores1_1)
print('cv_scores1_3', cv_scores1_2)
print('cv_scores1_3', cv_scores1_3)
print('cv_scores2_1', cv_scores2_1)
print('cv_scores2_2', cv_scores2_2)
print('cv_scores2_3 ', cv_scores2_3 )

#
# In[9]:

#Test the data
print('Testing SVM Accuracy.....')

resultsY_11 = model1_1.predict(test_vectors1)
resultsY_21 = model2_1.predict(test_vectors1)

resultsY_12 = model1_2.predict(test_vectors2)
resultsY_22 = model2_2.predict(test_vectors2)

resultsY_13 = model1_3.predict(test_vectors3)
resultsY_23 = model2_3.predict(test_vectors3)

precision11, recall11, score11, support11 = metrics.precision_recall_fscore_support(Y_test,resultsY_11 )
precision21, recall21, score21, support21 = metrics.precision_recall_fscore_support(Y_test,resultsY_21 )
accuracy11 = metrics.accuracy_score(Y_test,resultsY_11)
accuracy21 = metrics.accuracy_score(Y_test,resultsY_21)
confusionmatrix11=metrics.confusion_matrix(Y_test,resultsY_11)
confusionmatrix21=metrics.confusion_matrix(Y_test,resultsY_21)

precision12, recall12, score12, support12 = metrics.precision_recall_fscore_support(Y_test,resultsY_12 )
precision22, recall22, score22, support22 = metrics.precision_recall_fscore_support(Y_test,resultsY_22 )
accuracy12 = metrics.accuracy_score(Y_test,resultsY_12)
accuracy22 = metrics.accuracy_score(Y_test,resultsY_22)
confusionmatrix12=metrics.confusion_matrix(Y_test,resultsY_12)
confusionmatrix22=metrics.confusion_matrix(Y_test,resultsY_22)

precision13, recall13, score13, support13 = metrics.precision_recall_fscore_support(Y_test,resultsY_13 )
precision23, recall23, score23, support23 = metrics.precision_recall_fscore_support(Y_test,resultsY_23 )
accuracy13 = metrics.accuracy_score(Y_test,resultsY_13)
accuracy23 = metrics.accuracy_score(Y_test,resultsY_23)
confusionmatrix13=metrics.confusion_matrix(Y_test,resultsY_13)
confusionmatrix23=metrics.confusion_matrix(Y_test,resultsY_23)


print('__________________________________________') 
print('RESULTS - Model 1 - OneVsRestClassifier(LinearSVC())') 
print('__________________________________________') 

print ('BINARY')
print('Accuracy', accuracy11)
print('F1_score', score11)
print ('COUNT')
print('Accuracy', accuracy12)
print('F1_score', score12)
print ('TFIDF')
print('Accuracy', accuracy13)
print('F1_score', score13)
print('__________________________________________') 
#print('__________________________________________') 
print('RESULTS - Model 2 - OneVsOneClassifier(LinearSVC())') 
print('__________________________________________') 
print ('BINARY')
print('Accuracy', accuracy21)
print('F1_score', score21)
print ('COUNT')
print('Accuracy', accuracy22)
print('F1_score', score22)
print ('TFIDF')
print('Accuracy', accuracy23)
print('F1_score', score23)
print('__________________________________________') 
print('__________________________________________') 


## =============================================================================
## Apply Vader analysis on test set
## ============================================================================= 
vader_scores_test=[]
vader_scores_test_mod=[]
analyzer = SentimentIntensityAnalyzer() # create a vader analyzer
for tweet in X_test:
    vader_scores_test.append(analyzer.polarity_scores(tweet)['compound'])#apply analyser to tweet set

#convert scores to positive, neg and neutral, 0.3 was chosen due to the compound score being between 1 and neg 1
for score in vader_scores_test:
    if float(score)>=0.30: #positive sentiment
        vader_scores_test_mod.append(4)
    elif float(score)<=-0.30: #negativesentiment
        vader_scores_test_mod.append(0)
    else: 
        vader_scores_test_mod.append(2)
    
precision_vader, recall_vader, score_vader, support_vader = metrics.precision_recall_fscore_support( Y_test,vader_scores_test_mod)
accuracy_vader = metrics.accuracy_score( Y_test,vader_scores_test_mod)
confusionmatrix_vader=metrics.confusion_matrix( Y_test,vader_scores_test_mod)

print('RESULTS - VADER') 
print('__________________________________________') 
print('Accuracy', accuracy_vader)
print('F1_score',score_vader)
print('__________________________________________') 
print('__________________________________________') 

    
#    
### =============================================================================
### Apply best model to twitter data and print a selection of results
### =============================================================================
# NOTE: Current best model is Model1_3 One vs rest tdif vectorizer, with current parameters
file = open('olympics.json','r')
text = file.read()
split_text = text.split('\n')
split_text.pop() #Last elemnt is empty because of the split

tweet_test = [json.loads(data)['text'] for data in split_text]

# In[65]:

tweet_vector = vectorizer3.transform(tweet_test)
tweet_prediction = model1_3.predict(tweet_vector)


# In[68]:

#print 2 random tweets of each predicted class 
neutral_ind=(tweet_prediction==2).nonzero()[0]
pos_ind=(tweet_prediction==4).nonzero()[0]
neg_ind=(tweet_prediction==0).nonzero()[0]

tweet_test=numpy.asarray(tweet_test)
neutral=tweet_test[neutral_ind]
pos=tweet_test[pos_ind]
neg=tweet_test[neg_ind]
tweet_test=tweet_test.tolist()

rand_smpl_neutral = [ neutral[i] for i in sorted(random.sample(range(len(neutral)), 2)) ]
rand_smpl_pos = [ pos[i] for i in sorted(random.sample(range(len(pos)), 2)) ]
rand_smpl_neg = [ neg[i] for i in sorted(random.sample(range(len(neg)), 2)) ]
print('__________________________________________') 
print('___________PREDICTIONS_______________') 
print('__________________________________________') 

print('__________________________________________') 
print('NEUTRAL')
print(rand_smpl_neutral)
print('__________________________________________') 
print('POSITIVE')
print(rand_smpl_pos)
print('__________________________________________') 
print('NEGATIVE')
print(rand_smpl_neg)

## =============================================================================
## Apply Vader analysis and compare scores, <=-0.8: #negativesentiment >=0.8: #positive sentiment
## ============================================================================= 
vader_scores_orig=[]
vader_scores=[]

analyzer = SentimentIntensityAnalyzer() # create a vader analyzer
for tweet in tweet_test:
    vader_scores_orig.append(analyzer.polarity_scores(tweet)['compound'])#apply analyser to tweet set

#convert scores to positive, neg and neutral 
for score in vader_scores_orig:
    if float(score)>=0.8: #positive sentiment
        vader_scores.append(4)
    elif float(score)<=-0.8: #negativesentiment
        vader_scores.append(0)
    else: 
        vader_scores.append(2)
    
## =============================================================================
## Analyse results
## ============================================================================= 

precisionT, recallT, scoreT, supportT = metrics.precision_recall_fscore_support( vader_scores,tweet_prediction )
accuracyT = metrics.accuracy_score( vader_scores,tweet_prediction)
confusionmatrixT=metrics.confusion_matrix( vader_scores,tweet_prediction)

print('TWEET- Model-Vader comparison') 
print('__________________________________________') 
print('Accuracy', accuracyT)
print('F1_score', scoreT)
print('__________________________________________') 
print('__________________________________________') 










