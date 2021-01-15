import json
import math
#import pandas as pd

# =============================================================================
# UniqueWord: list
# bagOfWord: list
# uniqueWordCount: dict => output
# =============================================================================
# =============================================================================
# def countWords(UniqueWord,bagOfWords):
#     uniqueWordsCount = dict.fromkeys(UniqueWord, 0)
# 
#     # count number of time a word appear in all the news
#     for word in bagOfWords:
#         uniqueWordsCount[word] += 1
# 
#     return uniqueWordsCount
# =============================================================================

## input: news[ID] = [title, #0 #lst of str
##                    text,  #1 #lst of str
##                    uniqueTitleWord, #2 #lst of str
##                    uniqueTextWord   #3 #lst of str
##                    numOfWord] #4 #dict of (str, int)
def getTFs(news):    
    for ID in news.keys():
        title = news[ID][0]
        text = news[ID][1]
        numOfWord = news[ID][4]
        
        bagOfWords = text + title
    
        tf = computeTF(numOfWord, len(bagOfWords))
        news[ID].append(tf)
        
    return news

## output: tfDict #dict of (str,float)
def computeTF(wordDict, wordCount):
    tfDict = {}
    
    for word, count in wordDict.items():
        tfDict[word] = count / float(wordCount)
    return tfDict

## input: news[ID] = [title, #0 #lst of str
##                     text,  #1 #lst of str
##                     uniqueTitleWord, #2 #lst of str
##                     uniqueTextWord   #3 #lst of str
##                     numOfWord, #4 #dict of (str, int)
##                     tf] #5 #dict of (str,float)
## output: dict of (str,float)
def getIDF(news):
    documents = []
    for ID in news.keys():
        documents.append(news[ID][4]) #numOfWord
    return computeIDF(documents)

## input: lst of dict of (str,int): [{str: int, ...}, ...]
## output: idfDict # dict of (str,float)
def computeIDF(documents):
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:        
        for word, val in document.items():
            if word in idfDict:
                if val > 0:
                    idfDict[word] += 1
            else:
                idfDict[word] = 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def getTFIDF(news, idf):
    for ID in news.keys():
        tfidf = computeTFIDF(news[ID][5], idf)
        news[ID].append(tfidf)
    return news

## input: tfBagOfWords: dict of (str,float)
##       idfs: dict of (str,float)
## output: tfidf: dict of (str,float)
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def saveJSON(myDict,fileNname):
    j = json.dumps(myDict)
    with open(fileNname,'w') as f:
        f.write(j)
        f.close()
    return

## datatype: news = [title, #0 #lst of str
##                  text,   #1 #lst of str
##                  uniqueTitleWord, #2 #lst of str
##                  uniqueTextWord]  #3 #lst of str

f1 = open('fakeNews.json', encoding='utf-8')
fakeNews = json.load(f1)
f1.close()

f2 = open('realNews.json', encoding='utf-8')
realNews = json.load(f2)
f2.close()

f3 = open('uniquefake.json', encoding='utf-8')
UNIQUE_Fake = json.load(f3)
f3.close()

f4 = open('uniquereal.json', encoding='utf-8')
UNIQUE_Real = json.load(f4)
f4.close()

        
# get words and their occurence in each news
for ID in fakeNews.keys():
    numOfWord = dict.fromkeys(UNIQUE_Fake, 0)
    
    # count words apperance in title
    for word in fakeNews[ID][0]:
        numOfWord[word] += 1
    
    # count words apperance in title
    for word in fakeNews[ID][1]:
        numOfWord[word] += 1
    
    fakeNews[ID].append(numOfWord)

for ID in realNews.keys():
    numOfWord = dict.fromkeys(UNIQUE_Real, 0)
    
    # count words apperance in title
    for word in realNews[ID][0]:
        numOfWord[word] += 1
    
    # count words apperance in title
    for word in realNews[ID][1]:
        numOfWord[word] += 1
    
    realNews[ID].append(numOfWord)

## UPDATE: news = [title, #0 #lst of str
##                 text,  #1 #lst of str
##                 uniqueTitleWord, #2 #lst of str
##                 uniqueTextWord   #3 #lst of str
##                 numOfWord] #4 #dict of (str, int)

# calculate TF
# allText include title and text
FakeNews = getTFs(fakeNews)
RealNews = getTFs(realNews) 

## UPDATE: news = [title, #0 #lst of str
##                 text,  #1 #lst of str
##                 uniqueTitleWord, #2 #lst of str
##                 uniqueTextWord   #3 #lst of str
##                 numOfWord, #4 #dict of (str, int)
##                 tf] #5 #dict of (str,float)

j = json.dumps(FakeNews)
with open('tf_fake.json','w') as f:
    f.write(j)
    f.close()
    
j = json.dumps(RealNews)
with open('tf_real.json','w') as f:
    f.write(j)
    f.close()
    
# calculate IDF
IDF_fake_news = getIDF(FakeNews) # dict of (str,float)
IDF_real_news = getIDF(RealNews) # dict of (str,float)

j = json.dumps(IDF_fake_news)
with open('IDF_fake_news.json','w') as f:
    f.write(j)
    f.close()
    
j = json.dumps(IDF_real_news)
with open('IDF_real_news.json','w') as f:
    f.write(j)
    f.close()
    
## STOP HERE
### calculate TF-IDF
##Fake = getTFIDF(FakeNews, IDF_fake_news)
##Real = getTFIDF(RealNews, IDF_real_news)
##
#### UPDATE: news = [title, #0 #lst of str
####                 text,  #1 #lst of str
####                 uniqueTitleWord, #2 #lst of str
####                 uniqueTextWord   #3 #lst of str
####                 numOfWord, #4 #dict of (str, int)
####                 tf, #5 #dict of (str,float)
####                 tfidf] #6 #dict of (str,float)
##
##j = json.dumps(Fake)
##with open('TFIDF_fake_news.json','w') as f:
##    f.write(j)
##    f.close()
##    
##j = json.dumps(Real)
##with open('TFIDF_real_news','w') as f:
##    f.write(j)
##    f.close()
    
# create DF
##myData = []
##for ID in FakeNews.keys():
##    myData.append(FakeNews[ID][5])
##    
##df = pd.DataFrame(myData)
##df.to_csv(r'.\tf_fake.csv')
