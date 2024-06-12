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


def comb_tags_func(x):
    list_tags = x.values
    tag_comb = ' '.join(list_tags)
    return tag_comb

# Tags per question id
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

def tag_cooccurrence(tag_df):
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
    return tag_df


def tags_per_user(tag_df, q_df, a_df):
    # Create a list of all tags
    all_tags = set()
    for tags in tag_df['tag']:
        for tag in tags:
            all_tags.add(tag)
    all_tags = list(all_tags)

    # Create a dictionary of tags per user
    tags_per_user = {}
    for user in a_df['OwnerUserId'].unique():
        # if str(user) != 'nan':
        tags_per_user[str(user)] = {tag: 0 for tag in all_tags}
    # also add question users:
    for user in q_df['OwnerUserId'].unique():
        # if str(user) != 'nan':
        tags_per_user[str(user)] = {tag: 0 for tag in all_tags}
    
    for index, row in tag_df.iterrows():
        question_id = row['Id']
        tags = row['tag']
        users = a_df[a_df['ParentId'] == question_id]['OwnerUserId']
        users = users.dropna().values
        for user in users:
            for tag in tags:
                # if str(user) != 'nan':
                tags_per_user[str(user)][tag] += 1
        # Also add the user that asked the question
        user = q_df[q_df['Id'] == question_id]['OwnerUserId'].values[0]
        # if str(user) != 'nan':
        for tag in tags:
            tags_per_user[str(user)][tag] += 1
    
    return pd.DataFrame(tags_per_user).T 


def create_heatmap(matrix, title, xaxis, yaxis):
    # Create the heatmap
    heatmap = go.Heatmap(
        x=matrix.columns,
        y=matrix.index,
        z=matrix.values
    )
    # Create the layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
        height=800,
        width=2000
    )
    # Create the figure and add the heatmap
    heatmap = go.Figure(data=[heatmap], layout=layout)
    return heatmap

nrows = 2000

a_df = pd.read_csv('data/Answers.csv', encoding='latin-1', nrows=nrows)
q_df = pd.read_csv('data/Questions.csv', encoding='latin-1', nrows=nrows)
tags = pd.read_csv('data/Tags.csv', encoding='latin-1', nrows=nrows)

tag_df = create_tags(tags, q_df)
# tag_matrix = tag_cooccurrence(tag_df)
# tag_heatmap = create_heatmap(tag_matrix)

tags_per_user_df = tags_per_user(tag_df, q_df, a_df)

# Saving/reading dataset
# tags_per_user_df.to_csv('data/tags_per_user_df_2000.csv', index=True)
# tags_per_user_df = pd.read_csv('data/tags_per_user_df_2000.csv', index_col=0)
# tags_per_user_df.index = tags_per_user_df.index.astype(str)

tags_per_user_hm = create_heatmap(
    tags_per_user_df,
    title='Tags per user',
    xaxis='Tags',
    yaxis='Users'
)

print("DONE REFRESHING")

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='Simple Dash app with stackoverflow Q&A dataset'),
    html.Hr(),
    # dcc.Graph(figure=tag_heatmap, id='matrix')
    dcc.Graph(figure=tags_per_user_hm, id='tags-per-user'),
    html.Div(children={}, id='show-question')
]

# Create callback func to show sample question when a item in the graoh is clicked
@app.callback(
    Output('show-question', 'children'),
    Input('tags-per-user', 'clickData')
)
def show_sample_question(clickData):
    if clickData is None:
        return 'Click on a cell to show a sample question'
    userid = clickData['points'][0]['y']
    tag = clickData['points'][0]['x']
    print(userid, tag)
    print(q_df.columns)
    user_questions = q_df[q_df['OwnerUserId'] == float(userid)]
    tag_questions = user_questions[user_questions['tag'].apply(lambda x: tag in x)]
    if tag_questions.empty:
        parent_ids = a_df[a_df['OwnerUserId'] == float(userid)]
        user_questions = q_df[q_df['Id'].isin(parent_ids['ParentId'])]
        tag_questions = user_questions[user_questions['tag'].apply(lambda x: tag in x)]
        
    if tag_questions.empty:
        return 'No questions found for this user'
    

    # question = tag_questions.sample(1)
    question_text = tag_questions['Body'].values[0]
    question_title = tag_questions['Title'].values[0]
    return [
        html.H1(f"User: {userid}, Tag: {tag}"),
        html.H3(f"Title: {question_title}"),
        html.H3(f"Body: {question_text}")
    ]
    # return f"User: {userid}, Tag: {tag}\nTitle: {question_title}\n\nBody: {question_text}"

print(a_df.columns)
# @callback(
#     Output('show-question', 'figure'),
#     Input('tags-per-user', 'selectedData')
# )
# def show_sample_question(selectedData):
#     userid = selectedData['y']
#     tag = selectedData['x']
#     user_questions = q_df[q_df['OwnerUserId'] == int(userid)]
#     tag_questions = user_questions[user_questions['tag'].apply(lambda x: tag in x)]
#     question = tag_questions.sample(1)
#     question_text = question['Body'].values[0]
#     question_title = question['Title'].values[0]
#     return f"Title: {question_title}\n\nBody: {question_text}"


if __name__ == '__main__':
    app.run(debug=True)