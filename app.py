import pandas as pd 
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from preprocessing import create_tags, tags_per_user, create_heatmap
from constants import ALLOWED_TYPES

nrows = 2000

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

heatmap_versions = [tags_per_user_hm]

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
                        "Click on a cell to show a sample question",
                        id='show-question',
                        fluid=True, className='question'
                    ),
                    className='question-container'
                ),
            ],
            fluid=True, className='main-container'
        )
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
    Output('version', 'options'),
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
    # tags_per_user_df = normalize_df(tags_per_user_df)
    tags_per_user_hm = create_heatmap(
        tags_per_user_df,
        title='Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )

    heatmap_versions.append(tags_per_user_hm)

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
    Input('version', 'value')
)
def update_heatmap(version):
    if version is None:
        return [
            heatmap_versions[-1],
            f"Current version: Version {len(heatmap_versions)}"
        ]
    return [
        heatmap_versions[int(version.split(' ')[1])-1],
        f"Current version: {version}"
    ]


if __name__ == '__main__':
    app.run(debug=True)