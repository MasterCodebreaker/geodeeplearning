from vistools import *

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from graph_regression import SimpleGIN, GINLayer, unit_test_mini_batch, create_mini_batch, Graph, train_eval, GIN
train_zinc_dataset = ZINC(root='', split='train', subset=True)
val_zinc_dataset = ZINC(root='', split='val', subset=True)
test_zinc_dataset = ZINC(root='', split='test', subset=True)

# @title [RUN] Hyperparameters GIN 

BATCH_SIZE = 128 #@param {type:"integer"}
NUM_EPOCHS =   30 #@param {type:"integer"}
HIDDEN_DIM =   64 #@param {type:"integer"}
LR         = 0.001 #@param {type:"number"}

#you can add more here if you need


if __name__ == "__main__":
    """ 
    one_graph = train_zinc_dataset[0]
    #gallery([one_graph], labels=np.array([one_graph.y]), max_fig_size=(8,10))

    print(f"\nTrain examples: {len(train_zinc_dataset)}")
    print(f"Val examples: {len(val_zinc_dataset)}")
    print(f"Test examples: {len(test_zinc_dataset)}\n")


    print(f"First graph contains {one_graph.x.shape[0]} nodes, each characterised by {one_graph.x.shape[1]} features")
    print(f"Graph labels have shape: {one_graph.y.shape}")

    print(f'First graph : {train_zinc_dataset[0].x.shape} with adjacency {(train_zinc_dataset[0].num_nodes, train_zinc_dataset[0].num_nodes)}')
    print(f'Second graph: {train_zinc_dataset[1].x.shape} with adjacency {(train_zinc_dataset[1].num_nodes, train_zinc_dataset[1].num_nodes)}')
    print(f'Third graph : {train_zinc_dataset[2].x.shape} with adjacency {(train_zinc_dataset[2].num_nodes, train_zinc_dataset[2].num_nodes)}')
    #@title Visualize the mini-batching for a small list of batch_size=3 graphs.
    # Note that the three graphs viusalized are directed, 
    # so the adjacency matrix will be non-symmetric 
    # (even if the visualisation depicted them as undirected)

    # 3 random custom-designed graphs for visualisations
    graph1 = Graph(x=torch.rand((3,32)), 
                   y=torch.rand((1)), 
                   edge_index=torch.tensor([[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]]))
    graph2 = Graph(x=torch.rand((5,32)), 
                   y=torch.rand((1)), 
                   edge_index=torch.tensor([[0,1,2,3,4,0],[0,1,2,3,4,4]]))#[[0,0,0,0,0,1,1,1,2,1,2,3,4], [0,1,2,3,4,2,3,4,4,0,0,0,0]]))
    graph3 = Graph(x=torch.rand((4,32)),
                   y=torch.rand((1)), 
                  edge_index=torch.tensor([[0,1,2,3],[0,1,2,3]]))
    list_graphs = [graph1, graph2, graph3]

    # create a mini-batch from these 3 graphs
    batch_sample = create_mini_batch(list_graphs)

    # show statistics about the new graph built from this batch of graphs
    print(f"Batch number_of_nodes: {batch_sample.num_nodes}")
    print(f"Batch features shape: {batch_sample.x.shape}")
    print(f"Batch labels shape: {batch_sample.y.shape}")

    print(f"Batch adjacency: ")
    print_color_numpy(batch_sample.get_adjacency_matrix().to_dense().numpy(), list_graphs)

    gallery([graph1, graph2, graph3, batch_sample], max_fig_size=(20,6), special_color=True)
    print(f"And we also have access to which graph each node belongs to {batch_sample.batch}\n")
    
    

    array = torch.tensor([13, 21, 3, 7, 11, 20, 2])
    index = torch.tensor([0,1,1,0,2,0,1])

    aggregate_sum = scatter_sum(array, index, dim=0)
    aggregate_mean = scatter_mean(array, index, dim=0)
    aggregate_max, aggregate_argmax = scatter_max(array, index, dim=0)

    print("Let's inspect what different scatter functions compute: ")
    print(f"sum aggregation: {aggregate_sum}")
    print(f"mean aggregation: {aggregate_mean}")
    print(f"max aggregation: {aggregate_max}\n")

    batch_zinc = create_mini_batch(train_zinc_dataset[:3])
    # ============ YOUR CODE HERE =============
    # Given the nodes features for a batch of graphs (batch_zinc.x) 
    # and the list of indices indicating what graph each node belongs to
    # apply scatter_* to obtain a graph embedings for each graph in the batch
    # You can play with all of them (scatter_mean/scatter_max/scatter_sum)
    #print(batch_zinc.x.shape, batch_zinc.batch.shape)
    #exit()
    node_emb = batch_zinc.x[:,0] #scatter_mean(batch_zinc.x[:,0], batch_zinc.batch)
    node_batch = batch_zinc.batch
    graph_emb = scatter_sum(node_emb, node_batch)
    # ==========================================
    print(node_emb)
    print(node_batch)
    print(graph_emb)
    



    #@title Run unit test for mini-batch implementation
    #batch = train_zinc_dataset[:BATCH_SIZE]
    #unit_test_mini_batch(batch)

    batch_zinc = create_mini_batch(train_zinc_dataset[:3])
    # Instantiate our GIN model
    model_simple_gin = SimpleGIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)
    out, _ = model_simple_gin(batch_zinc)
    print(out.detach().numpy())


    #Train GIN model:
    train_stats_simple_gin_zinc = train_eval(model_simple_gin, train_zinc_dataset, val_zinc_dataset, 
                              test_zinc_dataset, loss_fct=F.mse_loss, 
                              metric_fct=F.mse_loss, print_every=150)
    plot_stats(train_stats_simple_gin_zinc, name='Simple_GIN_ZINC', figsize=(5, 10))


    """

    batch_zinc = create_mini_batch(train_zinc_dataset[:3])
    #@title Run unit test for mini-batch implementation
    #batch = train_zinc_dataset[:BATCH_SIZE]
    #unit_test_mini_batch(batch)

    model_gin = GIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)
    out, _ = model_gin(batch_zinc)
    print(out.detach().numpy())

    #Train GIN model:
    train_stats_gin_zinc = train_eval(model_gin, train_zinc_dataset, val_zinc_dataset, 
                                      test_zinc_dataset, loss_fct=F.mse_loss, 
                                      metric_fct=F.mse_loss, print_every=150)
    plot_stats(train_stats_gin_zinc, name='GIN_ZINC', figsize=(5, 10))
