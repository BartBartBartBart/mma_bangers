import dash_bootstrap_components as dbc



def create_help_popup():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("How to use")),
            dbc.ModalBody('In this project, we visualize hypergraphs that represent a stack overflow dataset of users and the tags/topics that they post questions on.'),
            dbc.ModalBody('Our target users are Stack Overflow contributors who are experts in their respective fields and use the platform to enhance their professional visibility.'
                          'They can interact with a hypergraph message passing model by updating the likelihood of one particular user posting a question related to a particular tag, and visualize how the model updates all other usertag pairing values via message passing.'
                          'Our system is relevant to these users as it allows them to strategically learn about new fields/topics based on the current trends on Stack Overflow, thereby enabling them to answer more questions related to emerging topics. '),
            dbc.ModalBody('Buttons', style={"font-weight":"bold"}),
            dbc.ModalBody('i. Deselect: Deselects the selected cell '),
            dbc.ModalBody('ii. Preview Changes: '),
            dbc.ModalBody('iii. Save Changes: After editing the cells, a new version can be saved by clicking on saved changes.'),
        ],
        id="help-popup",
        is_open=False,
    )
