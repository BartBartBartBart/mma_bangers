import dash_bootstrap_components as dbc



def create_help_popup():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("How to use")),
            dbc.ModalBody('In this project, we visualize hypergraphs that represent a stack overflow dataset of users and the tags/topics that they post questions on.'),
            dbc.ModalBody('Our target users are Stack Overflow contributors who are experts in their respective fields and use the platform to enhance their professional visibility. '
                          'They can interact with a hypergraph message passing model by updating the likelihood of one particular user posting a question related to a particular tag. ' 
                          'The model is finetuned on the new hypergraph and then predicts the likelihood of other user-tag combinations. ' 
                          'Our system is relevant to these users as it allows them to strategically learn about new fields/topics based on the current trends on Stack Overflow, ' 
                          'thereby enabling them to answer more questions related to emerging topics. '),
            dbc.ModalBody('In addition to an interactive hypergraph visualization, we allow users to save and compare versions of the hypergraph after making changes. ' 
                          'Finally, we visualize the embeddings of the users learned by the model in a 2D space using UMAP.'),
            dbc.ModalBody('Buttons', style={"font-weight":"bold"}),
            dbc.ModalBody('i. Implement Changes: Edits the selected cell with the new value.'),
            dbc.ModalBody('ii. Preview Changes: Shows the changes to would result by editing the selected cell with the selected value. These changes are not directly implemented.'),
            dbc.ModalBody('iii. Deselect: Deselects the selected cell.'),
            dbc.ModalBody('iv. Show Manual Edits: Shows the manual edits made by the user. The changes made by the model are not shown.'),
            dbc.ModalBody('v. Save Changes: Saves the changes made to the hypergraph'),
        ],
        id="help-popup",
        is_open=False,
    )
