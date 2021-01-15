import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDF(data):
    # TFIDF for FAKE
    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(data)
    
    df = pd.DataFrame(tfIdf[0].T.todense(), \
                      index=tfIdfVectorizer.get_feature_names(), \
                          columns=["TFIDF"])
    
    df = df.sort_values('TFIDF', ascending=False)
    
    #drop words with 0 point TFIDF
    #df = df.drop(df[df['TFIDF'] <= 0].index) 
    df.reset_index(level=0, inplace=True)
    
    #dataframe into dictionary
    myDict = df.set_index('index').T.to_dict('list')
    
    return myDict

def finalScore(f,r):
    
    f_dict = TFIDF(f)
    r_dict = TFIDF(r)
    
    
    # COMBINE TFIDF SCORE
    for word, point in f_dict.items():
        if word in r_dict:
            r_dict[word] = r_dict[word][0] - point[0]
        else:
            r_dict[word] = 0 - point[0]
    for word, point in r_dict.items():
        if type(r_dict[word]) is list:
            r_dict[word] = point[0]
    
    df = pd.DataFrame(r_dict.items(), columns=['Word', 'TFIDF'])
    df.to_csv('TFIDF.csv', index = False)
        
    return r_dict

import random
import nltk
from nltk.corpus import stopwords

def categorize(string):
    score = 0
    
    for word in string:
        if word in TF_IDF:
            score += TF_IDF[word]
            
    if score < 0:
        return 0
    return 1

def shuffling():    
    choices = list(range(0,len(data)))
    print(choices[-1])
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
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    garbage = "~`!@#$%^&*()_-+={[}]|\:;'<,>.?/"
    
    for i in range(dataset.shape[0]):
        #print(i)
        string = str(dataset.iloc[[i],[0]])
        #print('['+string+']')
        #tokenize
        string1 = tokenizer.tokenize(string)
        #remove stop word
        string2 = ([token.lower() for token in string1 \
                               if token not in stop_words])
        #normalize
        string3 = " ".join([lemmatizer.lemmatize(token) for token \
                                    in string2]).strip()
        #remove garbage
        dataset.loc[i,'text'] = "".join([char for char in string3 \
                                   if char not in garbage])
    #print(dataset.columns)
    #print(dataset.head(10))
    return dataset

def fit_transform(dataset):
    # ignore terms that appear in less than 2 documents 
    # or appear in more than 50% of documents
    # also allow idf to overwrite weighting
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2),use_idf=True)
    features = tfidf.fit_transform(dataset["text"])
    return pd.DataFrame(features.todense(), columns = tfidf.get_feature_names())
    

def fit_corpus(train_data, test_data):
    corpus = pd.DataFrame({"text": train_data["text"]})
    corpus.text.append(test_data["text"], ignore_index=True)
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
    tfidf.fit(corpus["text"].values.astype('U'))
    return tfidf

def transform_data(tfidf, dataset):
    features = tfidf.transform(dataset["text"].values.astype('U'))
    return pd.DataFrame(features.todense(), columns = tfidf.get_feature_names())

data = pd.read_csv("clean-news.csv",encoding = "ISO-8859-1")
data['text'] = data['title'] + data['text']
data = data.drop('title',1)
data.loc[(data['label'] == 'FAKE'),'label'] = 0
data.loc[(data['label'] == 'REAL'), 'label'] = 1
#print(data.columns)
print(data.loc[data['text'].isnull()])
print('sample data:')
print(data.head())

indexList = []
shuffling()
trainIndex = getIndex([0,1,2,3])
testIndex = getIndex([4])
#print(testIndex)

trainData = data.iloc[trainIndex,:]
trainData = trainData[trainData['text'].notna()]

testData = data.iloc[testIndex,:]
testData = testData[testData['text'].notna()]

preprocess(trainData)
preprocess(testData)
print('trainData sample')
print(trainData.head())
print(trainData.shape)
print(trainData.label.value_counts())


fake = trainData.loc[(trainData['label'] == 0)]#),'text'])
real = trainData.loc[(trainData['label'] == 1)]#),'text'])

tfidf = fit_corpus(trainData, testData)  #Fitting the vecorizer

real_features = transform_data(tfidf, real)  #transforming 
fake_features = transform_data(tfidf, fake)    #Train and Test


real['label'].fillna(1, inplace=True)
fake['label'].fillna(1, inplace=True)
fake_labels = fake['label']  #Taking labels in separate
real_labels = real['label']    #variables

real_features = real_features.fillna(0)
fake_features = fake_features.fillna(0)

r_sum_column = real_features.sum(axis=0)
d = r_sum_column.to_frame()
d.reset_index(level=0, inplace=True)
d.rename(columns = {'index':'word', 0:'tfidf'}, inplace = True) 

f_sum_column = fake_features.sum(axis=0)
d1 = f_sum_column.to_frame()
d1.reset_index(level=0, inplace=True)
d1.rename(columns = {'index':'word', 0:'tfidf'}, inplace = True) 

d = d.merge(d1, on=('word'), suffixes=('_r', '_f'))
d['sum'] = d['tfidf_r'] - d['tfidf_f']
d = d.drop('tfidf_r', axis = 1)
d = d.drop('tfidf_f', axis = 1)
print(d.head(20))

myDict = d.set_index('word').T.to_dict('list')
TF_IDF = {}
for key,item in myDict.items():
    TF_IDF[key] = item[0]
    
correct = []
result = []
i = testData.index.tolist()
for index in i:
    correct.append(testData.loc[index,'label'])
    result.append(categorize(testData.loc[index,'text']))

df = pd.DataFrame(columns=['classification','correct'])
df['classification'] = result
df['correct'] = correct
df.to_csv('answers(1).csv', index = False) 

# get correct classification
correct_df = df.loc[(df['classification'] == df['correct'])]
#correct_df.to_csv('correctAns.csv', index = False)

print('percentage correct:', len(correct_df) / 1267 / len(testData))

# TFIDF guide
#https://programmerbackpack.com/tf-idf-explained-and-python-implementation/
#https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
#save dataframe
#https://datatofish.com/export-dataframe-to-csv/#:~:text=You%20can%20use%20the%20following%20template%20in%20Python,“%2C%20index%20%3D%20False%20”%20from%20the%20code%3A
