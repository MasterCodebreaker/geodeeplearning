from vistools import *
from SimpleGNN import SimpleGNN, train_eval_loop_gnn_cora 
from SimpleMLP import SimpleMLP, train_eval_loop 

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

class CoraDataset(object):
    def __init__(self):
        super(CoraDataset, self).__init__()
        cora_pyg = Planetoid(root='/tmp/Cora', name='Cora', split="full")
        self.cora_data = cora_pyg[0]
        self.train_mask = self.cora_data.train_mask
        self.valid_mask = self.cora_data.val_mask
        self.test_mask = self.cora_data.test_mask

    def train_val_test_split(self):
        train_x = self.cora_data.x[self.cora_data.train_mask]
        train_y = self.cora_data.y[self.cora_data.train_mask]

        valid_x = self.cora_data.x[self.cora_data.val_mask]
        valid_y = self.cora_data.y[self.cora_data.val_mask]

        test_x = self.cora_data.x[self.cora_data.test_mask]
        test_y = self.cora_data.y[self.cora_data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.cora_data.x

    def get_adjacency_matrix(self):
        # We will ignore this for the first part
        adj = to_dense_adj(self.cora_data.edge_index)[0]
        return adj


if __name__ == "__main__":
    # Lets download our cora dataset and get the splits
    cora_data = CoraDataset()
    train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

    # Always check and confirm our data shapes match our expectations
    print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
    print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
    print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")


    # Instantiate our model and optimiser
    A = cora_data.get_adjacency_matrix()
    X = cora_data.get_fullx()

    train_mask = cora_data.train_mask
    valid_mask = cora_data.valid_mask
    test_mask = cora_data.test_mask

    # MLP
    print("MPL task")
    # Instantiate our model 
    model = SimpleMLP(input_dim=train_x.shape[-1], output_dim=7)

    # Run training loop
    train_stats_mlp_cora = train_eval_loop(model, train_x, train_y, valid_x, valid_y, test_x, test_y)
    plot_stats(train_stats_mlp_cora, name="MLP_Cora")

    # simple GCN
    print("GCN task 1.3")
    
    model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=A)

    # Run training loop
    train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask, 
                                              X, valid_y, valid_mask, 
                                              X, test_y, test_mask
                                           )
    plot_stats(train_stats_gnn_cora, name="GNN_Cora")
