import torch
from topomodelx.nn.hypergraph.unigcn import UniGCN
from tqdm import tqdm
import pandas as pd


def indcidence_from_counts(counts):
    return torch.where(counts >= 0.5, 1, 0)


class PostCountPredictor(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.unigcn = UniGCN(in_channels=embedding_dim, hidden_channels=embedding_dim)
        embedding_dim = 32
        mlp = torch.nn.Sequential(
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
        
        return x_0_new, output_counts
        

def train(model, optimizer, criterion, epochs, x_0, incidence_1, target, finetune_idx=None):
    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        x_0_new, output_counts = model(x_0, incidence_1)

        if finetune_idx is None:
            loss = criterion(output_counts, target)
        else:
            loss = criterion(output_counts[finetune_idx], target[finetune_idx])

        loss.backward()
        optimizer.step()

        x_0 = x_0_new
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return model, x_0


if __name__ == "__main__":
    # TODO Train model and save trainable params in node_embeddings.pt and mlp_params.pt files

    # print(tags_per_user_df)

    # num_nodes = tags_per_user_df.shape[0]
    # x_0 = torch.nn.Embedding(num_nodes, 32)
    # target = torch.tensor(tags_per_user_df.to_numpy())
    # incidence_1 = torch.tensor(tags_per_user_df.to_numpy()) != 0
    # embedding_dim = 32

    # model = PostCountPredictor(embedding_dim)
    # model_parameters = list(model.parameters())
    # embedding_parameters = [x_0]
    # all_parameters = model_parameters + embedding_parameters
    # model = train(
    #     model=model,
    #     optimizer=torch.optim.Adam(all_parameters, lr=0.01),
    #     criterion=torch.nn.MSELoss(),
    #     epochs=100,
    #     x_0=x_0,
    #     incidence_1=incidence_1,
    #     target=target,
    # )