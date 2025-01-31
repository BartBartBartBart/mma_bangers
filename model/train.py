import torch
from topomodelx.nn.hypergraph.unigcn import UniGCN
from tqdm import tqdm
import pandas as pd
from data.preprocessing import get_tags_per_user
import wandb
import random
import numpy as np

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def indcidence_from_counts(counts):
    return torch.where(counts >= 0.5, 1, 0)


class PostCountPredictor(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.unigcn = UniGCN(in_channels=embedding_dim, hidden_channels=embedding_dim)
        embedding_dim = 32
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Linear(embedding_dim, 1),
        )

    def forward(self, x_0, incidence_1):
        x_0_new, x_1_new = self.unigcn(x_0, incidence_1)
        
        node_indices, hyperedge_indices = torch.nonzero(incidence_1, as_tuple=True)
        node_embeddings = x_0_new[node_indices]
        hyperedge_embeddings = x_1_new[hyperedge_indices]
        concatenated_embeddings = torch.cat((node_embeddings, hyperedge_embeddings), dim=1)

        mlp_outputs = torch.squeeze(self.mlp(concatenated_embeddings))
        output_counts = torch.zeros_like(incidence_1, dtype=torch.float)
        output_counts[node_indices, hyperedge_indices] = mlp_outputs
        
        return output_counts
        

def train(model, optimizer, criterion, epochs, x_0, incidence_1, target, finetune_idx=None, logging=False):
    set_seed(42)

    model.train()
    x_0.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        output_counts = model(x_0.weight, incidence_1)

        if finetune_idx is None:
            loss = criterion(output_counts, target)
        else:
            loss = criterion(output_counts[finetune_idx], target[finetune_idx])

        loss.backward()
        optimizer.step() # node embeddings are updated here

        if logging:
            # graph loss
            wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    model.eval()
    x_0.eval()
    return model, torch.nn.Embedding.from_pretrained(x_0.weight)


def finetune(pending_changes, tags_per_user_df, model, optimizer, x_0, clickData):
    # Finetuning
    # Map the User ID and Tag to Integer Indices
    user_id = clickData['points'][0]['y']  # This is the specific user ID
    user_idx = list(pending_changes[0].index).index(user_id)

    # Find the integer index for the tag
    tag = clickData['points'][0]['x']  # This is the tag (string)
    tag_idx = list(pending_changes[0].columns).index(tag)
    
    # model finetuning
    logging = False
    target = torch.tensor(pending_changes[0].to_numpy(), dtype=torch.float)
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
    predicted_df = pd.DataFrame(predicted_tags_per_user.detach().numpy(), index=tags_per_user_df.index, columns=tags_per_user_df.columns)

    # cap negative values to 0
    predicted_df[predicted_df < 0] = 0

    # Create list of changes to be made
    changes = []
    for user_id in pending_changes[0].index:
        for tag in pending_changes[0].columns:
            current_value = pending_changes[0].loc[user_id, tag]
            predicted_value = predicted_df.loc[user_id, tag]
            if current_value != predicted_value:
                changes.append([user_id, tag, current_value, predicted_value])

    return predicted_df, changes, x_0, model


if __name__ == "__main__":
    # TODO Train model and save trainable params in node_embeddings.pt and mlp_params.pt files
    tags_per_user_df = get_tags_per_user(data_dir="./data")
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