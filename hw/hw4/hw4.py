from bokeh.models import CustomJS, ColumnDataSource, Slider, CDSView, IndexFilter
from bokeh.models.widgets import Select
from bokeh.layouts import column, row, layout, widgetbox
from bokeh.plotting import figure, curdoc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split

#Prepare data
df = pd.read_csv('data.csv')
df = df.drop(['ID', 'belly'], axis = 1)
labels = df.loc[:, 'cancer':'Jewish'].columns.values

colors = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'brown'}
num_labels = {0:'No', 1:'Yes'}
numbers = {'No':0, 'Yes':1}

#Change label columns to numbers
df = df.replace(numbers)

#Train models
x = df.loc[:, ['SODAFREQ', 'BUNSFREQ']].as_matrix()
y = df.loc[:, 'diabetes'].as_matrix()

class_model = SGDClassifier(loss='hinge', penalty='l2', random_state=42)
class_model.fit(x, y)
class_labels = pd.DataFrame(class_model.predict(x))

clust_model = AgglomerativeClustering(n_clusters=2)
clust_labels = pd.DataFrame(clust_model.fit_predict(x))

#Create drop downs
select_x = Select(title='Select X axis:',
                  value='SODAFREQ',
                  width=200,
                  options=df.columns.values.tolist())

select_y = Select(title='Select Y axis:',
                  value='BUNSFREQ',
                  width=200,
                  options=df.columns.values.tolist())

select_feature = Select(title='Select Feature:',
                  value='diabetes',
                  width=200,
                  options=labels.tolist())

model_list = ["SGDClassifier", "LogisticRegression"]

select_model = Select(title='Select Model:',
                      value='SGDClassifier',
                      width=200,
                      options=model_list)
#Create data source
source = ColumnDataSource(data={'x': df.loc[:, 'SODAFREQ'], 'y': df.loc[:, 'BUNSFREQ'],
                                'feature_color': df.loc[:, 'diabetes'].map(colors),
                                'feature_label': df.loc[:, 'diabetes'],
                                'class_color': class_labels.loc[:, 0].map(colors),
                                'class_label': class_labels.loc[:, 0],
                                'clust_color': clust_labels.loc[:, 0].map(colors),
                                'clust_label': clust_labels.loc[:, 0]})

#Create plots
fig1=figure(title = 'Food Frequency Original')
fig1.circle('x', 'y', fill_color='feature_color', legend='feature_label', source=source, size = 10)
fig1.legend.location = 'bottom_right'

fig2=figure(title = 'Food Frequency Classification')
fig2.circle('x', 'y', fill_color='class_color', legend='class_label', source=source, size = 10)
fig2.legend.location = 'bottom_right'

fig3=figure(title = 'Food Frequency Clustering')
fig3.circle('x', 'y', fill_color='clust_color', legend='clust_label', source=source, size = 10)
fig3.legend.location = 'bottom_right'

#Establish callbacks
def update_axis(attrname, old, new):
    #Grab dropdown selections
    x_name = select_x.value
    y_name = select_y.value
    classifier = select_feature.value
    model = select_model.value

    #Get data for models
    x = df.loc[:, [x_name, y_name]].as_matrix()
    y = df.loc[:, classifier].as_matrix()

    #Train and test selected model with all records (too sparse for a train/test split)
    if model == 'SGDClassifier':
        class_model = SGDClassifier(loss='hinge', penalty='l2', random_state=42)
    elif model == 'LogisticRegression':
        class_model = LogisticRegression(random_state=42)

    class_model.fit(x, y)
    class_labels = pd.DataFrame(class_model.predict(x))

    clust_model = AgglomerativeClustering(n_clusters=2)
    clust_labels = pd.DataFrame(clust_model.fit_predict(x))

    #Update plot data source
    source.data = {'x': df.loc[:, x_name], 'y': df.loc[:, y_name],
                   'feature_color': df.loc[:, classifier].map(colors),
                   'feature_label': df.loc[:, classifier],
                   'class_color': class_labels.loc[:, 0].map(colors),
                   'class_label': class_labels.loc[:, 0],
                   'clust_color': clust_labels.loc[:, 0].map(colors),
                   'clust_label': clust_labels.loc[:, 0]}
                       
select_x.on_change('value', update_axis)
select_y.on_change('value', update_axis)
select_feature.on_change('value', update_axis)
select_model.on_change('value', update_axis)

#Display plots
layout = layout([
    row(
        column(row(fig1), widgetbox(select_x, select_y, select_feature)),
        column(row(fig2), widgetbox(select_model)),
        column(row(fig3))
    )
])

curdoc().add_root(layout)
curdoc().title = 'Food Frequency'

#Correlation Analysis
corr = df.corr()

#Diabetes
temp = corr.loc['diabetes', :]
print(temp[temp > 0.4])

#Cancer
temp = corr.loc['cancer', :]
print(temp[temp > 0.3])