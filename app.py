import pandas as pd
import torch
import wandb
from dash import Dash, dcc, html, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
from data.preprocessing import create_tags, tags_per_user, create_heatmap
from model.train import PostCountPredictor, train
import plotly.graph_objects as go
from constants import ALLOWED_TYPES

nrows = 25

a_df = pd.read_csv('data/Answers.csv', encoding='latin-1', nrows=nrows)
q_df = pd.read_csv('data/Questions.csv', encoding='latin-1', nrows=nrows)
tags = pd.read_csv('data/Tags.csv', encoding='latin-1', nrows=nrows)

tag_df = create_tags(tags, q_df)
# tag_matrix = tag_cooccurrence(tag_df)
# tag_heatmap = create_heatmap(tag_matrix)

tags_per_user_df = tags_per_user(tag_df, q_df, a_df)

tags_per_user_hm = create_heatmap(
    tags_per_user_df,
    title='Tags per user',
    xaxis='Tags',
    yaxis='Users'
)

print("DONE REFRESHING")

heatmap_versions = [tags_per_user_df.copy()]
# # try this
# temporal_heatmap = heatmap_versions[0] - heatmap_versions[1]
# # if this doesnt work, we can save the dataframes in addition or instead of the heatmaps themselves
# # and then recreate the heatmaps when needed
# # so like this (be sure to change it in the callback as well):
# heatmap_versions = [tags_per_user_df]
# # and then to make the heatmap:
# temporal_df = heatmap_versions[0] - heatmap_versions[1]
# temporal_heatmap = create_heatmap(
#     temporal_df,
#     title='Tags per user',
#     xaxis='Tags',
#     yaxis='Users'
# )
# # then put this temporal heatmap in dcc.graph

logging = False
num_nodes = tags_per_user_df.shape[0]
x_0 = torch.nn.Embedding(num_nodes, 32)
target = torch.tensor(tags_per_user_df.to_numpy(), dtype=torch.float)
incidence_1 = torch.zeros_like(target, dtype=torch.float)
incidence_1[target >= 1] = 1.0
embedding_dim = 32

model = PostCountPredictor(embedding_dim)
model_parameters = list(model.parameters())
embedding_parameters = list(x_0.parameters())
all_parameters = model_parameters + embedding_parameters
optimizer = torch.optim.Adam(all_parameters, lr=0.01)

if logging:
    wandb.login(key=open("WANDB_API_KEY.txt").readline().strip())
    wandb.init(project="hypergraph_visualization", config={"epochs": 100})
    wandb.watch(model)

model, x_0 = train(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.MSELoss(),
    epochs=100,
    x_0=x_0,
    incidence_1=incidence_1,
    target=target,
    logging=logging,
)

# torch.save(model.state_dict(), "saved_params/model_state_dict.pt")
# torch.save(x_0.weight, "saved_params/node_embeddings.pt")

if logging:
    wandb.finish()

# Initialize the app
app = Dash()

# App layout
app.layout = dbc.Container(
    [
        html.Div(
            children='StackOverflow Hypergraph Analysis',
            style={
                'textAlign': 'center',
                'fontSize': 20,
                'padding': 10
            }
        ),
        html.Hr(),
        dbc.Container(
            [
                dbc.Container(
                    [
                        dbc.Container(
                            [
                                html.P("Toolbar"),
                                dbc.Container(
                                    [
                                        dbc.Col(
                                            dcc.Input(
                                                id="input-number",
                                                type="number",
                                                placeholder="Input number",
                                                debounce=True,
                                            ),
                                            width=6,
                                            className='cell-input'
                                        ),
                                        dbc.Col(
                                            dbc.Col(
                                                html.Div("Click on a cell to edit.", 
                                                         id="edit-cell"),
                                                width=6,
                                                className='edit-cell'
                                            ),
                                            className='edit-cell-container'
                                        )
                                    ], 
                                    fluid=True, className='cell-input-container'
                                ),
                                dbc.Container(
                                    html.P("Currect version: Version 1",
                                            id='current-version'),
                                    fluid=True, className='current-version'
                                ),
                                dbc.Container(
                                    [
                                        html.Label('Select version:'),
                                        dcc.Dropdown(
                                            ["Version 1"], 
                                            id='version', 
                                            style={
                                                'width': '100%',
                                                'display': 'block',
                                            }
                                        )
                                    ],
                                    fluid=True,
                                    className='version-dropdown'
                                )
                            ],     
                            className='toolbar', fluid=True
                        ),
                        dbc.Container(
                            dbc.Container(
                                dcc.Graph(figure=tags_per_user_hm, id='tags-per-user'),
                                fluid=True, className='heatmap'
                            ),
                            fluid=True, className='heatmap-container'
                        ),
                    ], 
                    fluid=True, className='graph-toolbar-container'
                ),
                dbc.Col(
                    dbc.Container(
                        children="Click on a cell to show a sample question",
                        id='show-question',
                        fluid=True, className='question'
                    ),
                    className='question-container'
                ),
            ],
            fluid=True, className='main-container'
        ),
        dbc.Container(
            [
                html.P("Temporal Matrix"),
                dbc.Container(
                    dbc.Container(
                        dcc.Graph(figure=go.Figure(), id='temporal-matrix'),
                        fluid=True, className='heatmap'
                    ),
                    fluid=True, className='heatmap-container'
                )
            ],
            fluid=True, className='graph-toolbar-container'
        ),
    ], 
    fluid=True, className='top-level-container'
)

@app.callback(
    Output('show-question', 'children'),
    Input('tags-per-user', 'clickData')
)
def show_sample_question(clickData, q_df=q_df, a_df=a_df):
    if clickData is None:
        return 'Click on a cell to show a sample question'
    userid = clickData['points'][0]['y']
    tag = clickData['points'][0]['x']
    user_questions = q_df[q_df['OwnerUserId'] == float(userid)]
    tag_questions = user_questions[user_questions['tag'].apply(lambda x: tag in x)]
    if tag_questions.empty:
        parent_ids = a_df[a_df['OwnerUserId'] == float(userid)]
        user_questions = q_df[q_df['Id'].isin(parent_ids['ParentId'])]
        tag_questions = user_questions[user_questions['tag'].apply(lambda x: tag in x)]
        
    if tag_questions.empty:
        return 'No questions found for this user. Click on on another cell to show a sample question.'
    
    # question = tag_questions.sample(1)
    question_text = tag_questions['Body'].values[0]
    question_title = tag_questions['Title'].values[0]
    return [
        html.H3(f"User: {userid}, Tag: {tag}"),
        html.Div(children=f"Title: {question_title}"),
        html.Div(children=f"Body: {question_text}")
    ]


@app.callback(
    Output("edit-cell", "children"),
    Output("input-number", "value"),
    Output('tags-per-user', 'figure', allow_duplicate=True),
    Output('version', 'options', allow_duplicate=True),
    Output('current-version', 'children', allow_duplicate=True),
    Input('tags-per-user', 'clickData'),
    Input("input-number", "value"),
    prevent_initial_call=True
)
def edit_cell(clickData, input_number, tags_per_user_df=tags_per_user_df):
    # TODO: normalize df? 
    # TODO: preview changes before saving
    # TODO: save changes button instead? 
    if input_number is None or input_number == '':
        tags_per_user_hm = create_heatmap(
            tags_per_user_df,
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        current_val = tags_per_user_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']]
        return [html.Div(f"Clicked on cell: ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), current value: {current_val}"), 
                '', 
                tags_per_user_hm,
                [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
                'Current version: Version 1'
        ]

    # update heatmap with new value
    tags_per_user_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']] = input_number

    # Map the User ID and Tag to Integer Indices
    user_id = clickData['points'][0]['y']  # This is the specific user ID
    user_idx = list(tags_per_user_df.index).index(user_id)

    # Find the integer index for the tag
    tag = clickData['points'][0]['x']  # This is the tag (string)
    tag_idx = list(tags_per_user_df.columns).index(tag)

    # model finetuning
    logging = False
    target = torch.tensor(tags_per_user_df.to_numpy(), dtype=torch.float)
    finetune_idx = torch.tensor([user_idx, tag_idx])
    incidence_1 = torch.zeros_like(target, dtype=torch.float)
    incidence_1[target >= 0.5] = 1.0


    model, x_0 = train(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        epochs=50,
        x_0=x_0,
        incidence_1=incidence_1,
        target=target,
        finetune_idx=finetune_idx,
    )

    # model prediction
    model.eval()
    x_0.eval()
    predicted_tags_per_user = model(x_0.weight, incidence_1)
    tags_per_user_df = pd.DataFrame(predicted_tags_per_user.detach().numpy(), index=tags_per_user_df.index, columns=tags_per_user_df.columns)

    # tags_per_user_df = normalize_df(tags_per_user_df)
    tags_per_user_hm = create_heatmap(
        tags_per_user_df,
        title='Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )

    heatmap_versions.append(tags_per_user_df)

    return [
        html.Div(f"Editing cell ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), new value: {input_number}"),
        '',
        tags_per_user_hm,
        [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
         f"Current version: Version {len(heatmap_versions)}"
    ]

# add callback to update the heatmap with the selected version
@app.callback(
    Output('tags-per-user', 'figure'),
    Output('current-version', 'children'),
    Output('temporal-matrix', 'figure', allow_duplicate=True),
    Input('version', 'value'),
    prevent_initial_call=True
)
def update_heatmap(version):
    if version is None:
        version_index = len(heatmap_versions) - 1
    else:
        version_index = int(version.split(' ')[1]) - 1
    
    selected_df = heatmap_versions[version_index]
    tags_per_user_hm = create_heatmap(
        selected_df,
        title='Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )
    
    if version_index > 0:
        previous_df = heatmap_versions[version_index - 1]
        temporal_df = selected_df - previous_df
    else:
        # If it's the first version, compare with an empty DataFrame with the same shape
        temporal_df = selected_df - pd.DataFrame(0, index=selected_df.index, columns=selected_df.columns)
    
    temporal_heatmap = create_heatmap(
        temporal_df,
        title='Temporal Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )
    
    return [
        tags_per_user_hm,
        f"Current version: Version {version_index + 1}",
        temporal_heatmap
    ]


if __name__ == '__main__':
    app.run(debug=True)