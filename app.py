from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objects as go


# plot the data in histogram
def plot_hist(data, bins, xlabel, ylabel, title):
    plt.hist(data, bins = bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def get_message(Series: pd.Series):
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        del message_words[:15]
        result.iloc[row] = ''.join(message_words).strip()
    return result

def get_date(Series: pd.Series):
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        del message_words[0]
        del message_words[1:]
        result.iloc[row] = ''.join(message_words).strip()
        result.iloc[row] = result.iloc[row].replace('Date: ', '')
    print('Done parsing, converting to datetime format..')
    return pd.to_datetime(result)

def get_sender_and_receiver(Series: pd.Series):
    sender = pd.Series(index = Series.index)
    recipient1 = pd.Series(index = Series.index)
    recipient2 = pd.Series(index = Series.index)
    recipient3 = pd.Series(index = Series.index)

    for row,message in enumerate(Series):
        message_words = message.split('\n')
        sender[row] = message_words[2].replace('From: ', '')
        recipient1[row] = message_words[3].replace('To: ', '')
        recipient2[row] = message_words[10].replace('X-cc: ', '')
        recipient3[row] = message_words[11].replace('X-bcc: ', '')
        
    return sender, recipient1, recipient2, recipient3


def create_matrix(emails: pd.Series):
    # messages = pd.Series(index=emails.index)
    counts = {}
    for email in emails:
        message_words = email.split('\n')
        sender = message_words[2].replace('From: ', '')
        recipient = message_words[3].replace('To: ', '')
        if sender not in counts:
            counts[sender] = {}
        if recipient not in counts[sender]:
            counts[sender][recipient] = 1
        else:
            counts[sender][recipient] += 1
    return pd.DataFrame(counts).fillna(0)

def get_subject(Series: pd.Series):
    result = pd.Series(index = Series.index)
    
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        message_words = message_words[4]
        result[row] = message_words.replace('Subject: ', '')
    return result

def get_folder(Series: pd.Series):
    result = pd.Series(index = Series.index)
    
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        message_words = message_words[12]
        result[row] = message_words.replace('X-Folder: ', '')
    return result


emails = pd.read_csv('archive/emails.csv', delimiter=',', nrows=1000)
emails.dataframeName = 'emails.csv'

df = emails
df['text'] = get_message(df.message)
df['sender'], df['recipient1'], df['recipient2'], df['recipient3'] = get_sender_and_receiver(df.message)
df['Subject'] = get_subject(df.message)
df['folder'] = get_folder(df.message)
df['date'] = get_date(df.message)
matrix = create_matrix(df.message)
print('created matrix')
df = df.drop(['message', 'file'], axis = 1)

sender = df.sender.value_counts()
indices = sender.index
sender = pd.DataFrame(sender, columns=['count'])
sender['sender'] = indices

fig = px.bar(sender, x = 'count', y='sender', orientation = 'h', hover_data = [], height = 700, color = "sender")

# Create the heatmap
heatmap = go.Heatmap(
    x=matrix.columns,
    y=matrix.index,
    z=matrix.values
)
# Create the layout
layout = go.Layout(
    title='Email correspondence matrix',
    xaxis=dict(title='Sender'),
    yaxis=dict(title='Recipient'),
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
    dcc.Graph(figure=fig, id='graph'),
    dcc.Graph(figure=heatmap, id='matrix')
]


if __name__ == '__main__':
    app.run(debug=True)