#!/usr/bin/env python
# coding: utf-8

# # PREPROCESSING

# ## Importing the libraries

# In[1]:


import pandas as pd
import requests
import bs4 as bfs
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from textblob import TextBlob
import csv
import numpy as np


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')


# ## Importing the dataset

# In[3]:



read_file = pd.read_excel ("Output Data Structure.xlsx")

read_file.to_csv ("Output Data Structure.csv",
                index = None,
                header=True)

df = pd.DataFrame(pd.read_csv("Output Data Structure.csv"))
df = pd.read_csv('Output Data Structure.csv',index_col=0)
df


# ## Cleaning the dataset

# In[4]:


df = df.drop(columns = ['Unnamed: 15','Unnamed: 16','Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20' ])


# In[5]:


df


# # DATA EXTRACTION (WEB SCRAPING)

# In[6]:


li = [url for url in df['URL']]
li


# In[7]:


text = []
for url in li:
  text.append(requests.get(url,headers={"User-Agent": "XY"}))


# In[8]:



for i in range(len(text)):
  text[i] = bfs.BeautifulSoup(text[i].content,'html.parser')


# In[9]:


articles = []
for text in text:
  articles.append(text.find(attrs= {"class":"td-post-content"}).text)


# In[10]:


for i in range(len(articles)):
  articles[i]= articles[i].replace('\n','')


# # TEXT ANALYSIS

# In[11]:


stop_words = list(set(stopwords.words('english')))


# In[12]:


sentences = []
for article in articles:
  sentences.append(len(sent_tokenize(article)))


# In[13]:


cleaned_articles = [' ']*len(articles)


# In[14]:


for i in range(len(articles)):
  for w in stop_words:
    cleaned_articles[i]= articles[i].replace(' '+w+' ',' ').replace('?','').replace('.','').replace(',','').replace('!','')
    #? ! , .


# In[15]:


words = []
for article in articles:
  words.append(len(word_tokenize(article)))


# In[16]:


words_cleaned = []
for article in cleaned_articles:
  words_cleaned.append(len(word_tokenize(article)))


# In[18]:


file = open('C:\\Users\\Lenovo\\My datasets\\INTERNSHIP\\20211030 Test Assignment\\MasterDictionary\\positive-words.txt', 'r')
positive_words = file.read().split()
positive_words = [word.lower() for word in positive_words]


# In[19]:


file = open('C:\\Users\\Lenovo\\My datasets\\INTERNSHIP\\20211030 Test Assignment\\MasterDictionary\\negative-words.txt', 'r')
negative_words = file.read().split()


# ### Positive score

# In[21]:


positive_score = [0]*len(articles)
for i in range(len(articles)):
  for word in positive_words:
    for letter in cleaned_articles[i].lower().split(' '):
      if letter==word:
        positive_score[i]+=1


# ### Negative score

# In[22]:


negative_score = [0]*len(articles)
for i in range(len(articles)):
  for word in negative_words:
    for letter in cleaned_articles[i].lower().split(' '):
      if letter==word:
        negative_score[i]+=1


# In[23]:


print(positive_score)
print(positive_words)
print(negative_score)
print(negative_words)


# In[24]:


words_cleaned = np.array(words_cleaned)
sentences = np.array(sentences)


# In[25]:


df['POSITIVE SCORE'] = positive_score
df['NEGATIVE SCORE'] = negative_score


# In[26]:


df['POLARITY SCORE'] = (df['POSITIVE SCORE']-df['NEGATIVE SCORE'])/ ((df['POSITIVE SCORE'] +df['NEGATIVE SCORE']) + 0.000001)
polarity_score = df['POLARITY SCORE']


# In[27]:


polarity_score


# In[28]:


df['SUBJECTIVITY SCORE'] = (df['POSITIVE SCORE'] + df['NEGATIVE SCORE'])/( (words_cleaned) + 0.000001)


# In[29]:


df['AVG SENTENCE LENGTH'] = np.array(words)/np.array(sentences)


# In[30]:


complex_words = []
sylabble_counts = []


# In[31]:



for article in articles:
  sylabble_count=0
  d=article.split()
  ans=0
  for word in d:
    count=0
    for i in range(len(word)):
      if(word[i]=='a' or word[i]=='e' or word[i] =='i' or word[i] == 'o' or word[i] == 'u'):
           count+=1
#            print(words[i])
      if(i==len(word)-2 and (word[i]=='e' and word[i+1]=='d')):
        count-=1;
      if(i==len(word)-2 and (word[i]=='e' and word[i]=='s')):
        count-=1;
    sylabble_count+=count    
    if(count>2):
        ans+=1
  sylabble_counts.append(sylabble_count)
  complex_words.append(ans)           


# In[32]:


df['PERCENTAGE OF COMPLEX WORDS'] = np.array(complex_words)/np.array(words)


# In[33]:


df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])


# In[34]:


df['AVG NUMBER OF WORDS PER SENTENCE'] = df['AVG SENTENCE LENGTH']


# In[35]:


df['COMPLEX WORD COUNT'] = complex_words


# In[36]:


df['WORD COUNT'] = words


# In[37]:


df['SYLLABLE PER WORD'] = np.array(sylabble_counts)/np.array(words)


# In[38]:


total_characters = []
for article in articles:
  characters = 0
  for word in article.split():
    characters+=len(word)
  total_characters.append(characters)  


# In[39]:


personal_nouns = []
personal_noun =['I', 'we','my', 'ours','and' 'us','My','We','Ours','Us','And'] 
for article in articles:
  ans=0
  for word in article:
    if word in personal_noun:
      ans+=1
  personal_nouns.append(ans)


# In[40]:


df['PERSONAL PRONOUNS'] = personal_nouns
#as the all pronouns were cleared when clearing the stop words.


# In[41]:


df['AVG WORD LENGTH'] = np.array(total_characters)/np.array(words)


# In[ ]:


articles


# In[ ]:


cleaned_articles


# ## Creating output file

# In[45]:


df.to_csv (r'C:\Users\Lenovo\My datasets\INTERNSHIP\20211030 Test Assignment\output.csv', index = None, header=True)


# In[46]:


df


# In[43]:





# In[44]:





# # THANK YOU!

# In[ ]:





# In[ ]:





# In[ ]:




