import json
import math

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

# calculate TF-IDF
f1 = open('tf_fake.json', encoding='utf-8')
FakeNews = json.load(f1)
f1.close()

f2 = open('tf_real.json',encoding = 'utf-8')
RealNews = json.load(f2)
f2.close()

f3 = open('IDF_fake_news.json',encoding = 'utf-8')
IDF_fake_news = json.load(f3)
f3.close()

f4 = open('IDF_fake_news',encoding = 'utf-8')
IDF_real_news = json.load(f4)
f4.close()

Fake = getTFIDF(FakeNews, IDF_fake_news)
Real = getTFIDF(RealNews, IDF_real_news)

## UPDATE: news = [title, #0 #lst of str
##                 text,  #1 #lst of str
##                 uniqueTitleWord, #2 #lst of str
##                 uniqueTextWord   #3 #lst of str
##                 numOfWord, #4 #dict of (str, int)
##                 tf, #5 #dict of (str,float)
##                 tfidf] #6 #dict of (str,float)

j = json.dumps(Fake)
with open('TFIDF_fake_news.json','w') as f:
    f.write(j)
    f.close()
    
j = json.dumps(Real)
with open('TFIDF_real_news','w') as f:
    f.write(j)
    f.close()
    
