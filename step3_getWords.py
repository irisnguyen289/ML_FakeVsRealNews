import csv
import json
import random
import spacy

alphabet = "abcdefghijklmnopqrstuvwxyz"

REAL_DICT = {}
FAKE_DICT = {}
mydict= {}

UNIQUE_Fake = set()
UNIQUE_Real = set()

title = 0
text = 1
label = -1

StopWordFile = open('StopWord.txt')
words = StopWordFile.read()
StopWordFile.close()
StopWord = words.split(',')
#print(StopWord)

nlp = spacy.load("en_core_web_lg")

f = open('clean-news.csv', encoding='utf-8')
reader = csv.reader(f)

space = " "

ID = 1
for row in reader:
    # convert string of words to list of words
    Title = row[title].split(' ')
    
    if len(row) == 3:
        Text = row[text].split(' ')
    else:
        Text = ''

    while '' in Title:
        Title.remove('')
    while '' in Text:
        Text.remove('')

    # remove stop words
    index = 0
    while index < len(StopWord):
        while StopWord[index] in Title:
            Title.remove(StopWord[index])
        while StopWord[index] in Text:
            Text.remove(StopWord[index])
        index += 1
        
    # deal with unicode for title
    temp = []
    for word in Title:
        if len(word) == 1 and word in alphabet:
            continue
        
        if "\\x" == word[:2]:
            temp.append(word[2:])
        elif "\\u" == word[:2]:
            while "\\u" == word[:2]:
                word = word[6:]
                if len(word) == 0:
                    break
            if len(word) != 0:
                temp.append(word)
        else:
            temp.append(word)
            
    Title = temp[:]
    
    # deal with unicode for text
    tempo = []
    for word in Text:
        if len(word) == 1 and word in alphabet:
            continue
        
        #print(word)
        if "\\x" == word[:2]:
            tempo.append(word[2:])
        elif "\\u" == word[:2]:
            while "\\u" == word[:2]:
                word = word[6:]
                if len(word) == 0:
                    break
            if len(word) != 0:
                tempo.append(word)
        else:
            tempo.append(word)
            
    Text = tempo[:]
    
# =============================================================================
#     print(bow_title)
#     print(bow_text)
# =============================================================================
# =============================================================================
#     if row[label] == 'FAKE':
#         FAKE_DICT.append({
#             'Title': Title, 
#             'Text': Text, 
#             'uniqueTitleWord': list(bow_title), 
#             'uniqueTextWord': list(bow_text)
#         })    
# =============================================================================
    mywords = nlp(space.join(Title + Text))
    string = ''
    for w in mywords:
        string += w.lemma_ + ' '

    mydict[ID] = [string, row[label]]
    """ SKIP
    if row[label] == 'FAKE':
        FAKE_DICT[ID] = space.join(Title + Text)

    else: # REAL
        REAL_DICT[ID] = space.join(Title + Text)
        """
    ID += 1
    
##    DEBUG
# =============================================================================
#     if ID > 3:
#         break
# =============================================================================

f.close()

# save data
mydict.pop(1)
j = json.dumps(mydict)
with open('allNews_ready.json','w') as f:
    f.write(j)
    f.close()
"""
j = json.dumps(FAKE_DICT)
with open('fakeNews_ready.json','w') as f:
    f.write(j)
    f.close()

REAL_DICT.pop(0)
j = json.dumps(REAL_DICT)
with open('realNews_ready.json','w') as f:
    f.write(j)
    f.close()
"""

d = []

choices = list(range(2,len(mydict)+2))
print(choices[-1])
random.shuffle(choices)
count = 0
data = {}

while choices:
    count += 1
    index = choices.pop()
    data[index] = mydict[index]
    
    if count == len(mydict) / 5:
        d.append(data)
        count = 0
        data = {}

#DEBUG
for i in range(len(d)):
    print(len(d[i]))
    
for i in range(len(d)):
    name = 'file' + str(i) + '.json'
    
    j = json.dumps(d[i])
    with open(name,'w') as f:
        f.write(j)
        f.close()
    
#https://www.youtube.com/watch?time_continue=66&v=kdyIpmduhOM&feature=emb_logo
