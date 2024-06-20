import pandas as pd 
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from preprocessing import create_tags, tags_per_user, create_heatmap, normalize_df
from constants import ALLOWED_TYPES

nrows = 2000

a_df = pd.read_csv('data/Answers.csv', encoding='latin-1', nrows=nrows)
q_df = pd.read_csv('data/Questions.csv', encoding='latin-1', nrows=nrows)
tags = pd.read_csv('data/Tags.csv', encoding='latin-1', nrows=nrows)

tag_df = create_tags(tags, q_df)
# tag_matrix = tag_cooccurrence(tag_df)
# tag_heatmap = create_heatmap(tag_matrix)

tags_per_user_df = tags_per_user(tag_df, q_df, a_df)
tags_per_user_df = normalize_df(tags_per_user_df)

tags_per_user_hm = create_heatmap(
    tags_per_user_df,
    title='Tags per user',
    xaxis='Tags',
    yaxis='Users'
)

print("DONE REFRESHING")

edited_cells = []
heatmap_versions = [tags_per_user_df.copy()]
tags_per_user_df = normalize_df(tags_per_user_df)
pending_changes_df = tags_per_user_df.copy()

# create temporal matrix initialised with zeros
temporal_matrix = pd.DataFrame(0, index=tags_per_user_df.index, columns=tags_per_user_df.columns)
temporal_hm = create_heatmap(
    temporal_matrix,
    title='Temporal Tags per user',
    xaxis='Tags',
    yaxis='Users',
    colourscale=[[0, "red"], [0.5, "white"], [1, "green"]]
)

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
                                                min = 0,
                                                max = 1,
                                                step=0.1
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
                                    html.Button("Preview Changes", id="preview-changes", n_clicks=0),
                                    className='cell-input'
                                ),
                                dbc.Container(
                                    html.Button("Save Changes", id="save-changes", n_clicks=0),
                                    className='cell-input'
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
                dbc.Container(
                    [
                        html.P("Choose two versions to compare"),
                        dbc.Container(
                            [
                                html.Label('From:'),
                                dcc.Dropdown(
                                    ["Version 1"], 
                                    id='compare-version-1', 
                                    style={
                                        'width': '100%',
                                        'display': 'block',
                                    }
                                ),
                                # html.P("versus"),
                                html.Label('To:'),
                                dcc.Dropdown(
                                    ["Version 1"], 
                                    id='compare-version-2', 
                                    style={
                                        'width': '100%',
                                        'display': 'block',
                                    }
                                )
                            ],
                            fluid=True,
                            className='compare-version-dropdown'
                        )
                    ],     
                    className='temporal-toolbar', fluid=True
                ),
                dbc.Container(
                    dbc.Container(
                        dcc.Graph(figure=temporal_hm, id='temporal-matrix'),
                        fluid=True, className='heatmap'
                    ),
                    fluid=True, className='heatmap-container'
                )
            ],
            fluid=True, className='graph-toolbar-container'
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Preview Changes")),
                dbc.ModalBody(id="preview-content"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-preview", className="ms-auto", n_clicks=0)
                ),
            ],
            id="preview-modal",
            is_open=False,
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
    Output('version', 'options', allow_duplicate=True),
    Output('current-version', 'children', allow_duplicate=True),
    Output('compare-version-1', 'options', allow_duplicate=True),
    Output('compare-version-2', 'options', allow_duplicate=True),
    Input('tags-per-user', 'clickData'),
    Input("input-number", "value"),
    prevent_initial_call=True
)
def edit_cell(clickData, input_number, tags_per_user_df=tags_per_user_df):
    # TODO: normalize df? 
    # TODO: preview changes before saving
    # TODO: save changes button instead? 
    
    global pending_changes_df, edited_cells
    current_val = pending_changes_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']]

    if clickData is None:
        tags_per_user_hm = create_heatmap(
            pending_changes_df,
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        return [
            'Click on a cell to edit.', 
            '',
            tags_per_user_hm,
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
            'Current version: Version 1',
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))]
        ]
    if input_number is None or input_number == '':
        tags_per_user_hm = create_heatmap(
            pending_changes_df,
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        return [
            html.Div(f"Clicked on cell: ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), current value: {current_val}"), 
            '', 
            tags_per_user_hm,
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
            'Current version: Version 1',
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
            [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))]
        ]

    # update heatmap with new value
    pending_changes_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']] = input_number
    pending_changes_df = normalize_df(pending_changes_df)

    # track edited cells
    edited_cells.append((clickData['points'][0]['x'], clickData['points'][0]['y'], current_val, input_number))

    tags_per_user_hm = create_heatmap(
        pending_changes_df,
        title='Tags per user',
        xaxis='Tags',
        yaxis='Users'
    )

    return [
        html.Div(f"Editing cell ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), new value: {input_number}"),
        '',
        tags_per_user_hm,
        [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
        f"Current version: Version {len(heatmap_versions)}",
        [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))],
        [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))]
    ]

# add callback to update the heatmap with the selected version
@app.callback(
    Output('tags-per-user', 'figure'),
    Output('current-version', 'children'),
    # Output('temporal-matrix', 'figure', allow_duplicate=True),
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
    
    # if version_index > 0:
    #     previous_df = heatmap_versions[version_index - 1]
    #     temporal_df = selected_df - previous_df
    # else:
    #     # If it's the first version, compare with an empty DataFrame with the same shape
    #     temporal_df = selected_df - pd.DataFrame(0, index=selected_df.index, columns=selected_df.columns)
    
    # temporal_heatmap = create_heatmap(
    #     temporal_df,
    #     title='Temporal Tags per user',
    #     xaxis='Tags',
    #     yaxis='Users',
    #     colourscale='RdYlGn'
    # )
    
    return [
        tags_per_user_hm,
        f"Current version: Version {version_index + 1}",
        # temporal_heatmap
    ]

# add callback to compare two versions
@app.callback(
    Output('temporal-matrix', 'figure'),
    Input('compare-version-1', 'value'),
    Input('compare-version-2', 'value'),
    prevent_initial_call=True
)
def compare_versions(version1, version2):
    if version1 is None:
        version1_index = len(heatmap_versions) - 1
    else:
        version1_index = int(version1.split(' ')[1]) - 1
    
    if version2 is None:
        version2_index = len(heatmap_versions) - 1
    else:
        version2_index = int(version2.split(' ')[1]) - 1
    
    df1 = heatmap_versions[version1_index]
    df2 = heatmap_versions[version2_index]
    
    diff_df = normalize_df(df2 - df1)

    colorscale = [[0, "red"], [0.5, "white"], [1, "green"]]
    
    diff_heatmap = create_heatmap(
        diff_df,
        title='Difference between versions',
        xaxis='Tags',
        yaxis='Users',
        colourscale=colorscale,
        zmid=0
    )
    
    return diff_heatmap

@app.callback(
    Output("preview-modal", "is_open"),
    Output("preview-content", "children"),
    Input("preview-changes", "n_clicks"),
    Input("close-preview", "n_clicks"),
    State("preview-modal", "is_open"),
    prevent_initial_call=True
)
def preview_changes(n_clicks_preview, n_clicks_close, is_open):
    global edited_cells

    if n_clicks_preview > 0:
        if not edited_cells:
            changes_text = "No changes to show."
        else:
            changes_text = "\n".join([f"Edited cell: ({tag}, {userid}), old value: {old_val}, new value: {new_val}" 
                                      for tag, userid, old_val, new_val in edited_cells])
        return not is_open, html.Pre(changes_text)
    if n_clicks_close > 0:
        return not is_open, ""
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output('version', 'options', allow_duplicate=True),
    Output('current-version', 'children', allow_duplicate=True),
    Output('compare-version-1', 'options', allow_duplicate=True),
    Output('compare-version-2', 'options', allow_duplicate=True),
    Input('save-changes', 'n_clicks'),
    prevent_initial_call=True
)
def save_changes(n_clicks):
    global tags_per_user_df, pending_changes_df, heatmap_versions

    if n_clicks > 0:
        tags_per_user_df = pending_changes_df.copy()
        heatmap_versions.append(tags_per_user_df.copy())

        options = [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))]
        current_version = f"Current version: Version {len(heatmap_versions)}"

        return options, current_version, options, options
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)