import pandas as pd
import re
import nltk
import math
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering

#Read in data
nltk.download("stopwords")
data = pd.read_csv("data/lyrics.csv")
data.lyrics = data.lyrics.astype(str)

#Define metal words
metal_words = ['fire', 'hell', 'kill', 'slaughter', 'witch', 'scream', 'storm', 'darkness', 'fires', 'demon'
               'demons', 'devil', 'satan']

#Filter out only metal songs
data = data.loc[data.genre == "Metal", :]

#Parse the text, retaining only alphabetic characters and whitespace (new lines replaced with a space)
parse = lambda x : re.sub('[^a-zA-Z\s]', '', x).replace('\n', ' ').lower()

#Apply parsing function
data.lyrics = data.lyrics.apply(parse)

#Count all non-stopwords in a string
def wordcount(x):
    wordcount = {}
    stops = set(stopwords.words("english"))
    words = x.split()
    words = [word for word in words if word not in stops]
    for word in words:
        count = wordcount.get(word, 0)
        count = count + 1
        wordcount[word] = count
    return wordcount

#Create a column with wordcount dictionaries for each song
data["counts"] = data.lyrics.apply(wordcount)

#Create column with total words
data['total_words'] = data.counts.apply(lambda x: sum(x.values()))

#Remove songs with less than 10 words
data = data.loc[data['total_words'] >= 10, :]
data = data.reset_index()

def sorted_wc(temp):
    for w in sorted(temp, key=temp.get, reverse = True):
        print(w, temp[w])

def agg_wc(df):
    agg_count = {}
    for wc in df:
        agg_count.update(wc)
    return agg_count

def metalness(wc, metal_words):
    total_metal_words = 0
    total_words = sum(wc.values())
    for word in metal_words:
        count = wc.get(word, 0)
        total_metal_words += count
    return total_metal_words/total_words

metalness_apply = lambda x: metalness(x, metal_words)

#Establish metalness of each song
data['metalness'] = data['counts'].apply(metalness_apply)

clust_model = AgglomerativeClustering(n_clusters=2)
data['labels'] = pd.DataFrame(clust_model.fit_predict(data['metalness'].as_matrix().reshape(-1, 1)))

#Peform aggregations
by_artist = data.groupby('artist')['metalness', 'total_words'].agg(['mean']).sort_values('mean', ascending = False)