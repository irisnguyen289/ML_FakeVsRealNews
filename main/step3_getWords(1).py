'''
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
'''

import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import naive_bayes, svm

import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def shuffling():    
    choices = list(range(0,len(data)))
    print("Number of documents: ",choices[-1]) #check last index
    random.shuffle(choices)
    count = 0
    index = []
    
    while choices:
        count += 1
        index.append(choices.pop())
        
        if count == len(data) / 5:
            indexList.append(index)
            count = 0
            index = []

def getIndex(aList):
    index = []
    for cluster in aList:
        index += indexList[cluster]
    
    return index

def preprocess(dataset):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    garbage = "~`!@#$%^&*()_-+={[}]|\:;'<,>.?/"
    
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    for i in range(dataset.shape[0]):
        string = str(dataset.iloc[i,0])
        string = string.replace("\n", " ").replace("'s",' ')
        #print('remove newline\n',string)
        #tokenize
        string1 = word_tokenize(string)
        
        '''
        # 1st try of doing tokenizing
        #remove stop word
        string2 = ([token.lower() for token in string1 \
                               if token not in stop_words])
        #normalize
        string3 = " ".join([lemmatizer.lemmatize(token) for token \
                                    in string2]).strip()
        #remove garbage
        dataset.loc[i,'text'] = "".join([char for char in string3 \
                                   if char not in garbage])
        '''
        
        #2nd try
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(string1):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        
        dataset.loc[i,'text'] = ' '.join(Final_words)
        #print('aft \n',' '.join(Final_words))
        #break
    return dataset

def fit_transform(dataset):
    # ignore terms that appear in less than 2 documents 
    # or appear in more than 50% of documents
    # also allow idf to overwrite weighting
    tfidf = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1,2),use_idf=True)
    features = tfidf.fit_transform(dataset["text"])
    return pd.DataFrame(features.todense(), columns = tfidf.get_feature_names())
    

def fit_corpus(train_data, test_data):
    corpus = pd.DataFrame({"text": train_data["text"]})
    #corpus.text.append(test_data["text"], ignore_index=True)
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
    tfidf.fit(corpus["text"].values.astype('U'))
    return tfidf

def transform_data(tfidf, dataset):
    features = tfidf.transform(dataset["text"].values.astype('U'))
    return pd.DataFrame(features.todense(), columns = tfidf.get_feature_names())

df = pd.read_csv("clean-news.csv",encoding = "ISO-8859-1")
df['text1'] = df['title'] + ' ' + df['text']
df = df.drop('title',1)
df = df.drop('text',1)
df.loc[(df['label'] == 'FAKE'),'label'] = 0
df.loc[(df['label'] == 'REAL'), 'label'] = 1
#print('before', df.loc[0,'text1'])
d = pickle.loads(pickle.dumps(df))
d.rename(columns = {'text1':'text'}, inplace = True) 
data = d.reindex(columns=['text', 'label'])
#print('\nafter', data.loc[0,'text'])

'''
# DEGUB
print(data.columns)
print(data.loc[data['text'].isnull()])
print('sample data:')
print(data.head())
'''
indexList = []
shuffling()

data = data[data['text'].notna()]
data = data[data['label'].notna()]
data = preprocess(data)
data['label'].fillna(1, inplace=True)
data['text'].fillna('', inplace=True)
print('text' , data.loc[0,'text'])

x = data.iloc[:,0].values
y = data.iloc[:,1].values

td = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
X = td.fit_transform(x).toarray()
words = td.get_feature_names()
'''
import matplotlib.pyplot as plt
plot.figure(figsize=[20,4])
_ = plt.imshow(X)
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,\
                                                    test_size = 0.2,\
                                                    random_state = 0)
# Training the classifier & predicting on test data

'''
Logistic Regression
'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs')

model = classifier.fit(X_train, y_train)

y_pred = model.predict(X_test)

#print(model.predict_proba(X_test)) # [confident score of FAKE, REAL ]

#Coefficient of the features in the decision function.
print(model.coef_.shape)
coef = model.coef_.reshape(-1)
print("Number of Words: ", coef.shape[0])

#print(words[np.argmax(coef)])
idx = np.argsort(coef)[-10:] #get index of words that most indicate 'REAL'
print('10 words of REAL news')
for i in idx:
    print(words[i])
print("\n")

idx = np.argsort(coef)[:10] #get index of words that indicate 'FAKE'
print('10 words of FAKE news')
for i in idx:
    print(words[i])
print("\n")
    
# Classification metrics
from sklearn.metrics import accuracy_score, classification_report
classification_report = classification_report(y_test, y_pred)

print('======================================================')
print('\nLogistic Regession Accuracy: ', accuracy_score(y_test, y_pred)*100)
print('\nClassification Report')
print('\n', classification_report)


#####################################################################
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train,y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)
# Use accuracy_score function to get the accuracy
print('======================================================')
print("\nNaive Bayes Accuracy: ",accuracy_score(predictions_NB, y_test)*100)
print('\nClassification Report')
print('\n', classification_report)


#####################################################################
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train,y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy
print('======================================================')
print("\nSVM Accuracy: ",accuracy_score(predictions_SVM, y_test)*100)
print('\nClassification Report')
print('\n', classification_report)
