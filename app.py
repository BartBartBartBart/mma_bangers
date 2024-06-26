
import pandas as pd
import torch
import wandb
import dash
from dash import dcc, html,Input, Output, State
import dash_bootstrap_components as dbc
import widgets
import callbacks
import torch.nn as nn

from data.preprocessing import create_tags, tags_per_user, create_heatmap, create_embedding_fig
from model.train import PostCountPredictor, train, finetune
from constants import ALLOWED_TYPES

nrows = 2000
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

a_df = pd.read_csv('data/Answers.csv', encoding='latin-1', nrows=nrows)
q_df = pd.read_csv('data/Questions.csv', encoding='latin-1', nrows=nrows)
tags = pd.read_csv('data/Tags.csv', encoding='latin-1', nrows=nrows)

tag_df = create_tags(tags, q_df)

# We save the version in heatmap_version, which contains non-normalized dataframes
# We save current non-saved changes in pending_changes_df
# This is a list, where the first item is non-normalized df
# The second item is the normalized df
# heatmap_versions -> [non-normalized df version 1, non-normalized df version 2, ...]
# pending changes -> [current non-normalized df, current normalized df]

tags_per_user_df = tags_per_user(tag_df, q_df, a_df)
heatmap_versions = [tags_per_user_df.copy()]
pending_changes = [tags_per_user_df.copy()]
pending_changes.append(tags_per_user_df.copy())

tags_per_user_hm = create_heatmap(
    tags_per_user_df,
    title='Tags per user',
    xaxis='Tags',
    yaxis='Users'
)

edited_cells = []

# create temporal matrix initialised with zeros
temporal_matrix = pd.DataFrame(0, index=tags_per_user_df.index, columns=tags_per_user_df.columns)
temporal_hm = create_heatmap(
    temporal_matrix,
    title='Temporal Tags per user',
    xaxis='Tags',
    yaxis='Users',
    colourscale=[[0, "red"], [0.5, "white"], [1, "green"]]
)

# Model training
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

# create list of all tags
all_tags = tags_per_user_df.columns
# create UMAP of embeddings
umap_fig = create_embedding_fig(x_0)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

help_popup_widget = widgets.create_help_popup()

print("DONE REFRESHING")

# App layout
app.layout = dbc.Container(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            html.H4("StackOverflow Hypergraph Analysis"), 
                            className="ms-2"),
                            width={"size": 6, "offset": 3}
                        ),
                ], align="center", className="g-0", justify="center"
            ),
        ),
        html.Hr(),
        dbc.Container(
            [
                dbc.Container(
                    [
                        dbc.Row([
                            dbc.Col(
                                    html.P("Current Version: 1", id='current-version'),
                                ),
                            dbc.Col(
                                dbc.Stack([
                                    html.Label('Select Version:'),
                                    dcc.Dropdown(
                                            ["Version 1"], 
                                            id='version', 
                                            style={
                                                'width': '100%',
                                                'display': 'block',
                                            }
                                    )
                                ]),    
                            ),
                        ]),
                        dbc.Container(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Stack(
                                                [
                                                    html.Div(
                                                        [
                                                            "Edit a Cell ",
                                                            html.I(className="fas fa-question-circle fa-sm", id="tooltip-edit", style={"cursor":"pointer", "textAlign": "center"}),
                                                            dbc.Tooltip(
                                                                "Click on a cell to edit it",
                                                                target="tooltip-edit",
                                                                placement="top",
                                                            ),
                                                        ], id="edit-cell", className="text-muted"),
                                                    dcc.Input(
                                                        id="input-number",
                                                        type="number",
                                                        placeholder="Input number",
                                                        debounce=True,
                                                        min = 0,
                                                        step=0.1
                                                    ),
                                                    
                                                ],
                                            ),
                                        ),
                                    ], align="center", className="p-3"
                                ),
                                dcc.Store(id='preview-counter', data=0),
                                dcc.Store(id='close-preview-counter', data=0),
                                dcc.Store(id='implement-counter', data=0),
                                
                                dbc.Container([
                                    help_popup_widget,
                                    dbc.Stack([
                                        dbc.Button("Implement Changes", id="implement-changes", n_clicks=0, color="primary", className="me-1 m-2"),
                                        dbc.Button("Preview Changes", id="preview-changes", n_clicks=0, color="primary", className="me-1 m-2"),
                                        dbc.Button('Deselect', id='deselect-button', n_clicks=0, color="primary", className="me-1 m-2"),
                                        dbc.Button("Show Manual Edits", id="edit-backlog", n_clicks=0, color="primary", className="me-1 m-2"),
                                        dbc.Button("Save Changes", id="save-changes", n_clicks=0, color="success", className="me-1 m-2"),
                                        dbc.Button('Help', id='help-button', color="primary", className="me-1 m-2"),
                                    ], direction="horizontal"),
                                    
                                ], fluid=True, className="cell-input"),
                                
                            ],     
                            className='toolbar', fluid=True
                        ),
                        dbc.Container(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Container(
                                            children="Click on a cell to show a sample question",
                                            id='show-question',
                                            fluid=True, className='question', style={'height': '100%'}
                                        ), width=4, className='question-container'
                                    ),
                                    dbc.Col(
                                        dbc.Container(
                                            dcc.Graph(figure=tags_per_user_hm, responsive=True, id='tags-per-user', style={'height': '100vmin', 'width': '130vmin'}),
                                            fluid=True, className='heatmap', style={'overflow': 'auto'}
                                        ),
                                        width=8, className='heatmap-container'
                                    ),
                                ]
                            ),
                            
                        ), 
                        dbc.Container(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Relative Difference between Versions"),
                                                dbc.Container(
                                                    [
                                                        html.P('From:'),
                                                        dcc.Dropdown(
                                                            ["Version 1"], 
                                                            id='compare-version-1', 
                                                            style={
                                                                'width': '100px',
                                                                'display': 'block',
                                                            }
                                                        ),
                                                        html.P('To:'),
                                                        dcc.Dropdown(
                                                            ["Version 1"], 
                                                            id='compare-version-2', 
                                                            style={
                                                                'width': '100px',
                                                                'display': 'block',
                                                            }
                                                        )
                                                    ], className='compare-version-dropdown'
                                                ),
                                            ], className="temporal-toolbar"
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    dbc.Container(
                                        dcc.Graph(figure=temporal_hm, responsive=True, id='temporal-matrix', style={'height': '100vmin', 'width': '134vmin'}),
                                        fluid=True, className='heatmap-container', style={'maxHeight': '100%', 'maxWidth': '100%', 'overflow': 'auto'}
                                    ), 
                                )
                            ]
                        ),
                        dbc.Container(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.P("UMAP of embeddings"),
                                                dbc.Stack(
                                                    [
                                                        html.Div(
                                                            [
                                                                "Select Tag to Highlight: ",
                                                                html.I(className="fas fa-question-circle fa-sm", id="umap-tooltip", style={"cursor":"pointer", "textAlign": "center"}),
                                                                dbc.Tooltip(
                                                                    "Selecting a tag highlights the users that have that tag in red",
                                                                    target="umap-tooltip",
                                                                    placement="top",
                                                                ),
                                                            ], id="umap-text", className="text-muted"),
                                                        dcc.Dropdown(
                                                            all_tags, 
                                                            id='umap-tag', 
                                                            style={
                                                                'width': '50%',
                                                                'display': 'block',
                                                            }
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            className="temporal-toolbar"
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    dbc.Container(
                                        dcc.Graph(figure=umap_fig, id='umap-figure'),
                                        fluid=True, className='umap-container', style={'maxHeight': '100%', 'maxWidth': '100%', 'overflow': 'auto'}
                                    ), 
                                ),
                            ]
                        ),
                    ], 
                    
                ),
                
            ],
            fluid=True, className='main-container'
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
    # TODO add sampling 

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
    Output('implement-counter', 'data'),
    Output('tags-per-user', 'clickData'),
    Output('umap-figure', 'figure', allow_duplicate=True),
    Input('tags-per-user', 'clickData'),
    Input("deselect-button", "n_clicks"),
    Input("implement-changes", "n_clicks"),
    Input("implement-counter", "data"),
    State("input-number", "value"),
    prevent_initial_call=True
)
def edit_cell(clickData, deselect_clicks, implement_clicks, implement_counter, input_number, model=model, x_0=x_0, umap_fig=umap_fig):    
    global pending_changes, edited_cells

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    if ctx.triggered[0]['prop_id'] == 'deselect-button.n_clicks':
        pending_changes[0] = pending_changes[0].copy()
        
        tags_per_user_hm = create_heatmap(
            pending_changes[0], # latest df
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        return [
            html.Div(
                [
                    "Edit a Cell",
                    html.I(className="fas fa-question-circle fa-sm", id="tooltip-edit", style={"cursor":"pointer", "textAlign": "center"}),
                    dbc.Tooltip(
                        "Click on a cell to edit it",
                        target="tooltip-edit",
                        placement="top",
                    ),
                ], id="edit-cell", className="text-muted"
            ), 
            '',
            tags_per_user_hm,
            implement_clicks,
            None,
            umap_fig
        ]
    
    if clickData is None:
        tags_per_user_hm = create_heatmap(
            pending_changes[0], # latest df
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        return [
            'Click on a cell to edit.', 
            '',
            tags_per_user_hm,
            implement_clicks,
            clickData,
            umap_fig
        ]
    current_val = pending_changes[0].loc[clickData['points'][0]['y'], clickData['points'][0]['x']]

    if implement_clicks > implement_counter and input_number is not None and input_number != '':

        # update heatmap with new value
        updated_heatmap = pending_changes[0].copy()
        updated_heatmap.loc[clickData['points'][0]['y'], clickData['points'][0]['x']] = input_number
        pending_changes[0] = updated_heatmap

        # Finetuning
        # Map the User ID and Tag to Integer Indices
        user_id = clickData['points'][0]['y']  # This is the specific user ID
        user_idx = list(pending_changes[0].index).index(user_id)

        # Find the integer index for the tag
        tag = clickData['points'][0]['x']  # This is the tag (string)
        tag_idx = list(pending_changes[0].columns).index(tag)

        # model finetuning
        predicted_df, _, x_0 = finetune(
            pending_changes, 
            tags_per_user_df, 
            model, optimizer, 
            x_0,
            clickData
        )

        pending_changes[0] = predicted_df

        # update umap
        umap_fig = create_embedding_fig(x_0)
        
        # track edited cells
        edited_cells.append((clickData['points'][0]['x'], clickData['points'][0]['y'], current_val, input_number))

        tags_per_user_hm = create_heatmap(
            predicted_df, # latest normalized df
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )

        implement_counter += 1

        return [
            html.Div(f"Editing cell ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), new value: {input_number}"),
            '',
            tags_per_user_hm,
            implement_counter,
            clickData,
            umap_fig
        ]
    else: 
        tags_per_user_hm = create_heatmap(
            pending_changes[0], # latest df
            title='Tags per user',
            xaxis='Tags',
            yaxis='Users'
        )
        return [
            html.Div(f"Clicked on cell: ({clickData['points'][0]['x']}, {clickData['points'][0]['y']}), current value: {current_val}"), 
            input_number, 
            tags_per_user_hm,
            implement_clicks,
            clickData,
            umap_fig
        ]
    

@app.callback(
    Output('tags-per-user', 'figure'),
    Output('current-version', 'children'),
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
    
    return [
        tags_per_user_hm,
        f"Current version: Version {version_index + 1}",
    ]


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
    
    diff_df = df2 - df1

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
    Output('preview-counter', 'data'),
    Output('close-preview-counter', 'data'),
    Input("preview-changes", "n_clicks"),
    Input("close-preview", "n_clicks"),
    Input('tags-per-user', 'clickData'),
    Input("preview-counter", "data"),
    Input("close-preview-counter", "data"),
    State("input-number", "value"),
    prevent_initial_call=True
)
def preview_changes(n_clicks_preview, n_clicks_close, clickData, preview_counter, close_preview_counter, preview_number, model=model, x_0=x_0):
    # TODO: add fintuning
    # global edited_cells

    changes = []
    if clickData is not None and preview_number is not None and n_clicks_preview > preview_counter:
        preview_df = pending_changes[0].copy()
        preview_df.loc[clickData['points'][0]['y'], clickData['points'][0]['x']] = preview_number

        # model finetuning
        _, changes, _ = finetune(
            pending_changes, 
            tags_per_user_df, 
            model, optimizer, 
            x_0,
            clickData
        )

    if n_clicks_preview > preview_counter:
        if len(changes) == 0:
            changes_text = "No changes to show."
        else:
            changes_text = "\n".join([f"Edited cell: ({tag}, {userid}), old value: {old_val}, new value: {new_val}" 
                                      for tag, userid, old_val, new_val in changes])
        return True, html.Pre(changes_text),  n_clicks_preview, close_preview_counter
    if n_clicks_close > close_preview_counter:
        return False, "", preview_counter, n_clicks_close
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
    global heatmap_versions

    if n_clicks > 0:
        heatmap_versions.append(pending_changes[0].copy())

        options = [{'label': f"Version {i+1}", 'value': f"Version {i+1}"} for i in range(len(heatmap_versions))]
        current_version = f"Current version: Version {len(heatmap_versions)}"

        return options, current_version, options, options
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("preview-modal", "is_open", allow_duplicate=True),
    Output("preview-content", "children", allow_duplicate=True),
    Input("edit-backlog", "n_clicks"),
    Input("close-preview", "n_clicks"),
    State("preview-modal", "is_open"),
    prevent_initial_call=True
)
def edit_backlog(n_clicks_preview, n_clicks_close, is_open):
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
    Output('umap-figure', 'figure', allow_duplicate=True),
    Input('umap-tag', 'value'),
    prevent_initial_call=True
)
def highlight_tag(tag, all_tags=all_tags, x_0=x_0, pending_changes=pending_changes):
    if tag not in all_tags:
        return dash.no_update
    
    # embeddings are ordered the same as the users in pending_changes[0]
    # so get index of users in pending_changes[0] that have the selected tag
    tag_idx = []
    for idx, user in enumerate(pending_changes[0].index):
        if pending_changes[0].loc[user, tag] > 0:
            tag_idx.append(idx)

    # highlight with red the embeddings of users with the selected tag
    umap_fig = create_embedding_fig(x_0, highlight_idx=tag_idx)
    return umap_fig


# @app.callback(
#     Output('umap-figure', 'figure', allow_duplicate=True),
#     Input('implement-changes', 'n_clicks'),
#     Input('implement-counter', 'data'),
#     prevent_initial_call=True
# )
# def update_umap(implement_button_clicks, implement_counter, x_0=x_0):
#     if implement_button_clicks > implement_counter:
#         umap_fig = create_embedding_fig(x_0)
#         return umap_fig
#     raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run(debug=True)