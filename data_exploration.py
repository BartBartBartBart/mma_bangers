import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objects as go


nrows = 1000

a_df = pd.read_csv('data/Answers.csv', encoding='latin-1', nrows=nrows)
q_df = pd.read_csv('data/Questions.csv', encoding='latin-1', nrows=nrows)
tags = pd.read_csv('data/Tags.csv', encoding='latin-1', nrows=nrows)



def comb_tags_func(x):
    list_tags = x.values
    tag_comb = ' '.join(list_tags)
    return tag_comb


def create_tags(tags, q_df):
    temp = tags['Tag'].value_counts()
    temp = pd.DataFrame(temp)
    temp = temp.reset_index()
    temp.rename(columns = {'Counts':'Tag' , 'count':'Counts'}, inplace = True)
    tags = pd.merge(tags , temp , on = 'Tag')
    tags.dropna(inplace = True)
    tags = tags.sort_values(by = 'Id')
    tags['Id'] = tags['Id'].astype('int32')
    mapping_dict = tags.groupby('Id')['Tag'].apply(comb_tags_func)
    q_df['tag'] = q_df['Id'].map(mapping_dict)
    q_df.dropna(inplace=True)
    q_df['tag'] = q_df['tag'].apply(lambda x : x.split())
    return q_df

tag_df = create_tags(tags, q_df)

# get unique tags
unique_tags = set()
for tags in tag_df['tag']:
    for tag in tags:
        unique_tags.add(tag)
unique_tags = list(unique_tags)

# create a dictionary of tags that co-occur
tag_dict = {}
for tag in unique_tags:
    tag_dict[tag] = {}
    for tag2 in unique_tags:
        tag_dict[tag][tag2] = 0

for tags in tag_df['tag']:
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            tag_dict[tags[i]][tags[j]] += 1
            tag_dict[tags[j]][tags[i]] += 1

# create a dataframe from the dictionary
tag_df = pd.DataFrame(tag_dict)

# # create a graph from the dataframe
# G = nx.from_pandas_adjacency(tag_df)

# # create a layout for our nodes
# layout = nx.spring_layout(G, iterations=50)

# # draw the parts we want
# nx.draw_networkx_edges(G, layout, edge_color='#AAAAAA')

# # draw the most popular tags
# popular_tags = tag_df.sum().sort_values(ascending=False).index[:50]
# nx.draw_networkx_nodes(G, layout, nodelist=popular_tags, node_size=300, node_color='lightblue')

# # draw the labels
# nx.draw_networkx_labels(G, layout, font_size=10, font_family='sans-serif')

# # show the plot
# plt.title('Co-occurrence of tags')
# plt.axis('off')
# plt.show()


# Create the heatmap
heatmap = go.Heatmap(
    x=tag_df.columns,
    y=tag_df.index,
    z=tag_df.values
)
# Create the layout
layout = go.Layout(
    title='Tag Co-occurence matrix',
    xaxis=dict(title='Tag1'),
    yaxis=dict(title='Tag2'),
    height=800,
    width=1500
)
# Create the figure and add the heatmap
heatmap = go.Figure(data=[heatmap], layout=layout)

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='Simple Dash app with Elron email dataset'),
    html.Hr(),
    dcc.Graph(figure=heatmap, id='matrix')
]


if __name__ == '__main__':
    app.run(debug=True)