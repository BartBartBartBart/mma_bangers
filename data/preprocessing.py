import pandas as pd
import plotly.graph_objects as go
import umap
import os


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


def normalize_df(df):
    # Iterate over each user (row) in the DataFrame
    df = df.astype(float)
    for user in df.index:
        total = df.loc[user].sum()
        if total != 0:
            df.loc[user] = df.loc[user] / total
        else:
            df.loc[user] = 0
    return df



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
        # remove '.0' from the end of the user id
        user = str(user).split('.')[0]
        # if str(user) != 'nan':
        tags_per_user[str(user)] = {tag: 0 for tag in all_tags}
    # also add question users:
    for user in q_df['OwnerUserId'].unique():
        user = str(user).split('.')[0]
        # if str(user) != 'nan':
        tags_per_user[str(user)] = {tag: 0 for tag in all_tags}
    
    for index, row in tag_df.iterrows():
        question_id = row['Id']
        tags = row['tag']
        users = a_df[a_df['ParentId'] == question_id]['OwnerUserId']
        users = users.dropna().values
        # remove '.0' from the end of the user id
        users = [str(user).split('.')[0] for user in users]
        for user in users:
            for tag in tags:
                # if str(user) != 'nan':
                tags_per_user[str(user)][tag] += 1
        # Also add the user that asked the question
        user = q_df[q_df['Id'] == question_id]['OwnerUserId'].values[0]
        user = str(user).split('.')[0]
        # if str(user) != 'nan':
        for tag in tags:
            tags_per_user[str(user)][tag] += 1

    # Normalize all values to be between 0 and 1
    # tags_per_user = normalize_df(tags_per_user)
    
    # sort the dictionary alphabetically on tags
    # tags_per_user = {k: v for k, v in sorted(tags_per_user.items(), key=lambda item: item[0])}
    
    return pd.DataFrame(tags_per_user).T 


def create_heatmap(matrix, title, xaxis, yaxis, colourscale='Blues', zmid=None):
    # Create the heatmap
    heatmap = go.Heatmap(
        x=matrix.columns,
        y=matrix.index,
        z=matrix.values
    )
    # change colourscale to red to green
    # heatmap.colorscale = 'RdYlGn'
    # change colour scale to white to blue
    heatmap.colorscale = colourscale

    if zmid is not None:
        heatmap.zmid = zmid

    # Create the layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
        height=800,
        width=1400,
    )
    # Create the figure and add the heatmap
    heatmap = go.Figure(data=[heatmap], layout=layout)
    
    return heatmap

def create_embedding_fig(embeddings, highlight_idx=[], fit_transform=True):
    if fit_transform:
        reducer = umap.UMAP()
        umap_embeddings = reducer.fit_transform(embeddings.weight.data.numpy())
    else:
        umap_embeddings = embeddings
    umap_fig = go.Figure()
    blue_opacity = 1
    if len(highlight_idx) > 0:
        blue_opacity = 0.2
    for user_idx in range(umap_embeddings.shape[0]):
        if user_idx in highlight_idx:
            umap_fig.add_trace(go.Scatter(x=[umap_embeddings[user_idx,0]], y=[umap_embeddings[user_idx,1],], mode='markers', marker=dict(size=7, color='red')))
        else:
            umap_fig.add_trace(go.Scatter(x=[umap_embeddings[user_idx,0]], y=[umap_embeddings[user_idx,1],], mode='markers', marker=dict(size=7, color='blue', opacity=blue_opacity)))
    umap_fig.update_layout(title='UMAP of embeddings', xaxis_title='UMAP 1', yaxis_title='UMAP 2')
    umap_fig.update_layout(showlegend=False)

    # make the fig square
    umap_fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )

    # make the axes equal
    umap_fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # umap_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return umap_fig, umap_embeddings

def get_tags_per_user(data_dir, nrows=2000):
    a_df = pd.read_csv(os.path.join(data_dir, 'Answers.csv'), encoding='latin-1', nrows=nrows)
    q_df = pd.read_csv(os.path.join(data_dir, 'Questions.csv'), encoding='latin-1', nrows=nrows)
    tags = pd.read_csv(os.path.join(data_dir, 'Tags.csv'), encoding='latin-1', nrows=nrows)
    tag_df = create_tags(tags, q_df)
    tags_per_user_df = tags_per_user(tag_df, q_df, a_df)
    return tags_per_user_df
