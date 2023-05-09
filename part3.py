from vistools import *
from graph_regression import Graph, create_mini_batch, SimpleGIN
#@title [RUN] Hard to distinguish graphs
train_zinc_dataset = ZINC(root='', split='train', subset=True)
val_zinc_dataset = ZINC(root='', split='val', subset=True)
test_zinc_dataset = ZINC(root='', split='test', subset=True)
batch_zinc = create_mini_batch(train_zinc_dataset[:3])

# @title [RUN] Hyperparameters GIN 

BATCH_SIZE = 128 #@param {type:"integer"}
NUM_EPOCHS =   30#@param {type:"integer"}
HIDDEN_DIM =   64#@param {type:"integer"}
LR         = 0.001 #@param {type:"number"}

#you can add more here if you need

def gen_hard_graphs_WL():
  
  x1 = torch.ones((10,1))
  edge_index1 = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                 [2, 5, 1, 3, 2, 4, 6, 3, 5, 1, 4, 3, 7, 10, 6, 8, 7, 9, 8, 10, 6, 9]])-1
  y1 = torch.tensor([1])

  x2 = torch.ones((10,1))
  edge_index2 = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                 [2, 6, 1, 3, 7, 2, 4, 10, 3, 5, 4, 6, 1, 5, 2, 8, 7, 9, 8, 10, 3, 9]])-1
  y2 =  torch.tensor([2])  

  graph1 = Graph(x=x1, edge_index=edge_index1, y=y1)
  graph2 = Graph(x=x2, edge_index=edge_index2, y=y2)
  return [graph1, graph2]


if __name__ == "__main__":
    """
    hard_graphs = gen_hard_graphs_WL()
    #gallery(hard_graphs, labels=["A","B"], max_fig_size=(10,5))


    hard_batch = create_mini_batch(hard_graphs)
    model_simple_gin = SimpleGIN(input_dim=batch_zinc.x.size()[-1], output_dim=1, hidden_dim=HIDDEN_DIM, num_layers=4, eps=0.1)
    out, node_emb = model_simple_gin(hard_batch)

    #split node_emb from batch into separate graphs
    node_emb = node_emb.detach().numpy()
    node_emb_split=[node_emb[:hard_graphs[0].num_nodes], node_emb[hard_graphs[0].num_nodes:]]

    #encode node representation into an int in [0,1] denoting the color
    node_emb_split = hash_node_embedings(node_emb_split)


    gallery(hard_graphs, node_emb=node_emb_split, max_fig_size=(10,5))
    """
