import pandas as pd
import plotly.graph_objects as go
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


def normalize_df(tags_per_user):
    for user in tags_per_user:
        total = sum(tags_per_user[user].values())
        for tag in tags_per_user[user]:
            if total == 0:
                tags_per_user[user][tag] = 0
            else:
                tags_per_user[user][tag] /= total
    return tags_per_user


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

    # Normalize all values to be between 0 and 1
    # tags_per_user = normalize_df(tags_per_user)
    
    # sort the dictionary alphabetically on tags
    # tags_per_user = {k: v for k, v in sorted(tags_per_user.items(), key=lambda item: item[0])}
    
    return pd.DataFrame(tags_per_user).T 


def create_heatmap(matrix, title, xaxis, yaxis, colourscale='Blues'):
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


def get_tags_per_user(data_dir, nrows=2000):
    a_df = pd.read_csv(os.path.join(data_dir, 'Answers.csv'), encoding='latin-1', nrows=nrows)
    q_df = pd.read_csv(os.path.join(data_dir, 'Questions.csv'), encoding='latin-1', nrows=nrows)
    tags = pd.read_csv(os.path.join(data_dir, 'Tags.csv'), encoding='latin-1', nrows=nrows)
    tag_df = create_tags(tags, q_df)
    tags_per_user_df = tags_per_user(tag_df, q_df, a_df)
    return tags_per_user_df