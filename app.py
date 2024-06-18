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
                                                html.Div(id="edit-cell"),
                                                width=6,
                                                className='edit-cell'
                                            ),
                                            className='edit-cell-container'
                                        )
                                    ], 
                                    fluid=True, className='cell-input-container'
                                ),
                                dbc.Container(
                                    [
                                        html.Label('Select version:'),
                                        dcc.Dropdown(
                                            ["Version 1", "Version 2"], 
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
                    html.Div(
                        id='show-question',
                        children='Click on a cell to show a sample question',
                        # style={
                        #     'white-space': 'pre-wrap',
                        #     'width': '50%'
                        # }
                    ),
                    className='question-container'
                ),
            ],
            fluid=True, className='main-container'
        )
        # html.Div(
        #     [
        #         dcc.Input(
        #             id="input-number",
        #             type=_,
        #             placeholder=f"Input number",
        #             debounce=True,
        #         ) for _ in ALLOWED_TYPES
        #     ]
        # ),
        # html.Div(children='', id="edit-cell"),
        # html.Div(
        #     id='show-question',
        #     children='Click on a cell to show a sample question',
        #     style={
        #         'white-space': 'pre-wrap',
        #         'width': '50%'
        #     }
        # )
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
        return 'No questions found for this user'
    
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
    Output('tags-per-user', 'figure'),
    Input('tags-per-user', 'clickData'),
    Input("input-number", "value"),
)
def edit_cell(clickData, input_number, tags_per_user_df=tags_per_user_df):
    if clickData is None:
        tags_per_user_hm = create_heatmap(
            tags_per_user_df,
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        # value = tags_per_user_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']]
        return ["Click on a cell to edit", '', tags_per_user_hm]
    elif input_number is None or input_number == '':
        tags_per_user_hm = create_heatmap(
            tags_per_user_df,
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        current_val = tags_per_user_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']]
        return [html.Div(f"Clicked on cell: ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), current value: {current_val}"), 
                '', 
                tags_per_user_hm]
    # update heatmap with new value
    tags_per_user_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']] = input_number
    # tags_per_user_df = normalize_df(tags_per_user_df)
    tags_per_user_hm = create_heatmap(
        tags_per_user_df,
        title='Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )
    return [
        html.Div(f"Editing cell ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), new value: {input_number}"),
        '',
        tags_per_user_hm
    ]

if __name__ == '__main__':
    app.run(debug=True)