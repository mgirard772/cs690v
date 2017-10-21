import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bokeh.models import CustomJS, ColumnDataSource, Slider, CDSView, IndexFilter, HoverTool
from bokeh.models.widgets import Select, RadioGroup
from bokeh.layouts import column, row, layout, widgetbox
from bokeh.plotting import figure, curdoc
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import OrderedDict

#Read in data
nltk.download("stopwords")
data = pd.read_csv("data/lyrics.csv")
data.lyrics = data.lyrics.astype(str)

#Define metal words
metal_words = ['fire', 'hell', 'kill', 'slaughter', 'witch', 'scream', 'storm', 'darkness', 'fires', 'demon'
               'demons', 'devil', 'satan', 'mountains', 'hellfire', 'fireball', 'vikings', 'viking', 'ritual',
               'burn', 'cries', 'veins', 'eternity', 'breathe', 'beast', 'gonna', 'ashes', 'soul', 'sword',
               'sorrow', 'pray', 'reign', 'flames']

#Filter out only metal songs
data = data.loc[data.genre == "Metal", :]

#Parse the text, retaining only alphabetic characters and whitespace (new lines replaced with a space)
parse = lambda x : re.sub('[^a-zA-Z\s]', '', x).replace('\n', ' ').lower()

#Apply parsing function
data.lyrics = data.lyrics.apply(parse)

#Count all non-stopwords in a string
def wordcount(x, stops):
    wordcount = {}
    words = x.split()
    words = [word for word in words if word not in stops]
    for word in words:
        count = wordcount.get(word, 0)
        count = count + 1
        wordcount[word] = count
    return wordcount

#Create a column with wordcount dictionaries for each song
stops = set(stopwords.words("english"))
data["counts"] = data.lyrics.apply(lambda x: wordcount(x, stops))

#Create some features
data['total_words'] = data.counts.apply(lambda x: sum(x.values()))
data['unique_words'] = data.counts.apply(lambda x: x.__len__())


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

TOOLS="pan,wheel_zoom,box_zoom,reset,hover,previewsave"

artist_songs = data.groupby('artist').count()['song'].to_frame().reset_index()
artists_above_9 = sorted(artist_songs.artist[artist_songs.song > 9])
selected_artist = 'amon-amarth'
select_artist = Select(title='Select Artist:',
                  value=selected_artist,
                  width=200,
                  options=artists_above_9)

temp = data.loc[data.artist == selected_artist, :]
avg_metalness_year = temp.groupby('year')['metalness'].agg('mean').to_frame().reset_index()
source = ColumnDataSource(data={'year': temp.year,
                                'metalness': temp.metalness,
                                'artist': temp.artist,
                                'song': temp.song,
                                'unique_words': temp.unique_words,
                                'total_words': temp.total_words})
source_avg = ColumnDataSource(data={'avg_year': avg_metalness_year.year,
                                    'avg_metalness': avg_metalness_year.metalness})

#Non-clustering plots
fig1 = figure(title = 'Artist Song Metalness By Year', tools = TOOLS)
fig1.circle('year', 'metalness', source=source, size=10)
hover1 = fig1.select(dict(type=HoverTool))
hover1.tooltips = OrderedDict([
    ("Artist", "@artist"),
    ("Song", "@song"),
    ("Metalness", "@metalness{1.111}"),
    ("Year", "@year"),
])

fig2 = figure(title = 'Artist Song Metalness vs. Unique Words', tools = TOOLS)
fig2.circle('unique_words', 'metalness', source=source, size=10)
hover2 = fig2.select(dict(type=HoverTool))
hover2.tooltips = OrderedDict([
    ("Artist", "@artist"),
    ("Song", "@song"),
    ("Metalness", "@metalness{1.111}"),
    ("Unique Words", "@unique_words"),
    ("Year", "@year"),
])

fig3 = figure(title = 'Artist Average Metalness by Year', tools = TOOLS)
fig3.circle('avg_year', 'avg_metalness', source=source_avg, size=10)
fig3.line('avg_year', 'avg_metalness', source=source_avg)
hover3 = fig3.select(dict(type=HoverTool))
hover3.tooltips = OrderedDict([
    ("Year", "@avg_year"),
    ("Avg Metalness", "@avg_metalness{1.111}")
])

#Clustering Plots
colors = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'brown'}
select_artist_clust = Select(title='Select Artist:',
                  value=selected_artist,
                  width=200,
                  options=artists_above_9)
clust_feature = Select(value = 'metalness',
                       title = 'Select Clustering Feature:',
                       options = ['metalness', 'total_words', 'unique_words'],
                       width = 200)
slider_nclusters = Slider(value=2,
                         start=2,
                         end=5,
                         step=1,
                         title='Number of Clusters')
temp_clust = data.loc[data.artist == selected_artist, :]
model = AgglomerativeClustering(n_clusters=2)
labels = pd.DataFrame(model.fit_predict(temp_clust.metalness.as_matrix().reshape(-1, 1)), columns=['labels'])
labels['colors'] = labels['labels'].map(colors)
source_clust = ColumnDataSource(data={'year': temp_clust.year,
                                      'artist': temp_clust.artist,
                                      'unique_words': temp_clust.unique_words,
                                      'total_words': temp_clust.total_words,
                                      'song': temp_clust.song,
                                      'metalness': temp_clust.metalness,
                                      'labels': labels['labels'],
                                      'colors': labels['colors']})

fig4 = figure(title = 'Artist Song Metalness By Year, Clustered')
fig4.circle('year', 'metalness', fill_color = 'colors', source=source_clust, size=10)

fig5 = figure(title = 'Artist Song Metalness vs. Unique Words, Clustered')
fig5.circle('unique_words', 'metalness', fill_color = 'colors', source=source_clust, size=10)

fig6 = figure(title = 'Artist Song Metalness vs. Total Words, Clustered')
fig6.circle('total_words', 'metalness', fill_color = 'colors', source=source_clust, size=10)

# Establish callbacks
def update_artist(attrname, old, new):
    # Grab dropdown selection
    selected_artist = select_artist.value
    temp = data.loc[data.artist == selected_artist, :]
    avg_metalness_year = temp.groupby('year')['metalness'].agg('mean').to_frame().reset_index()
    source.data = dict(year=temp.year,
                       metalness=temp.metalness,
                       artist=temp.artist,
                       song=temp.song,
                       unique_words=temp.unique_words,
                       total_words=temp.total_words)
    source_avg.data = dict(avg_year=avg_metalness_year.year,
                           avg_metalness=avg_metalness_year.metalness)

def update_clustering(attrname, old, new):
    feature = clust_feature.value
    selected_artist = select_artist_clust.value
    clusters = slider_nclusters.value
    temp_clust = data.loc[data.artist == selected_artist, :]
    model = AgglomerativeClustering(n_clusters = clusters)
    labels = pd.DataFrame(model.fit_predict(temp_clust[feature].as_matrix().reshape(-1, 1)), columns=['labels'])
    labels['colors'] = labels['labels'].map(colors)
    source_clust.data = dict(year=temp_clust.year, artist=temp_clust.artist, unique_words=temp_clust.unique_words,
                             total_words=temp_clust.total_words, song=temp_clust.song, metalness=temp_clust.metalness,
                             labels=labels['labels'], colors=labels['colors'])

select_artist.on_change('value', update_artist)
select_artist_clust.on_change('value', update_clustering)
clust_feature.on_change('value', update_clustering)
slider_nclusters.on_change('value', update_clustering)

layout = layout([
    row(
        column(row(fig1), widgetbox(select_artist)),
        column(row(fig3)),
        column(row(fig2))
    ),
    row(
        column(row(fig4), widgetbox(select_artist_clust, clust_feature, slider_nclusters)),
        column(row(fig5)),
        column(row(fig6))
    )
])

curdoc().add_root(layout)
curdoc().title = 'Metal Lyrics Analysis'
