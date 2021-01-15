import csv
import string

myString = "0123456789abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
myNum = "0123456789"

punctuation = '''!()-–—‒+[]{};:'\,"«<>».z·•/?@#$%^&*_~â€™®°‘’“”…'''
table = str.maketrans(dict.fromkeys(string.punctuation))

f = open('news.csv','r', encoding ='utf-8')
reader = csv.reader(f)

content = ""

title = 1
text = 2
label = 3

ID = 0
previousIsText = True

for row in reader:
    Title = ""
    ttl = row[title]
    char = 0
    for char in ttl:
        if char.isnumeric():
            if previousIsText:
                Title += " "
            Title += char
            previousIsText = False
        elif char in myString:
            if (not previousIsText):
                Title += ' '
            Title += char
            previousIsText = True
        else:
            Title += ' '
            previousIsText = True
            

    Text = ""
    txt = row[text]
    previousIsText = True
    
    for char in txt:
        if char.isnumeric():
            if previousIsText:
                Text += " "
            Text += " "
            previousIsText = False
        elif char in myString:
            if (not previousIsText):
                Text += ' '
            Text += char
            previousIsText = True
        else:
            Title += ' '
            previousIsText = True
        
    Title = Title.lower().replace("\n"," ").replace("\r"," ")\
        .replace('æ','a').replace('á','a').replace('\u0092',' ')\
        .replace('\u0096',' ').replace('\u0091',' ').replace('ú','u')\
        .replace('\u0097',' ').replace('é','e').replace('u s ', 'u.s.')\
        .replace('\u00a0',' ').replace('\u00a9', ' ').replace('\ufeff',' ')\
        .replace('\u00ad','').replace('\u200b','').replace('\u2009','')\
        .replace('\u0093','').replace('\u0094','').replace('\u00ed','')

    Text = Text.lower().replace("\n", " ").replace("\r", " ")\
            .replace('æ','a').replace('á','a').replace('\u0092',' ')\
            .replace('\u0096',' ').replace('\u0091',' ').replace('ú','u')\
            .replace('\u0097',' ').replace('é','e').replace('u s ', 'u.s.')\
            .replace('\u00a0',' ').replace('\u00a9', ' ').replace('\ufeff',' ')\
            .replace('\u00ad','').replace('\u200b','').replace('\u2009','')\
            .replace('\u0093','').replace('\u0094','').replace('\u00ed','')\
            .replace('\u00a0',' ').replace('\u00a9', ' ')
    
    newrow = Title + "," + Text + "," + row[label] + "\n"
##    print(newrow)
    content += newrow

    # DEBUG
##    ID += 1
##    if ID > 11:
##        print(ttl)
##        print("NEW ROW:",Text)
##        break

file = open('clean-news.csv', 'w', encoding = 'utf-8')
file.write(content)
f.close()
file.close()
