# opening a file to read
StopWordList = open('stop-word-list.txt')
StopWord = StopWordList.readlines()
StopWordList.close()

# opening a file to write
file = open('StopWord.txt', 'w')
sw = ""

for word in StopWord:
    sw += word[:-1] + ','
    
file.write(sw[:-1])
file.close()
