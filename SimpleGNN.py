from vistools import *
from tqdm import tqdm
# @title [RUN] Hyperparameters GNN

NUM_EPOCHS =  100 #@param {type:"integer"}
LR         = 0.01 #@param {type:"number"}

#you can add more here if you need


# Fill in initialisation and forward method the GCNLayer below
class GCNLayer(nn.Module):
    """GCN layer to be implemented by students of practical

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # ============ YOUR CODE HERE =============
        # Compute symmetric norm
        A_t = self.A + torch.eye(A.size(dim=0))
        D_t = torch.zeros(self.A.size())
        for i in range(A_t.size(dim=0)):
            D_t[i,i] = torch.sum(A_t[i])

        # TODO: Note that the following is overkill, as D is already a diagonal matrix, we can just raise to power along diagonal.
        
        evals, evecs = torch.linalg.eig(D_t)  # get eigendecomposition
                                  # check decomposition
        evpow = evals**(-1/2)                              # raise eigenvalues to fractional power
        # build exponentiated matrix from exponentiated eigenvalues
        D_tp = torch.matmul(evecs, torch.matmul(torch.diag (evpow), torch.inverse (evecs))).real

        self.adj_norm = torch.matmul(D_tp, torch.matmul(A_t, D_tp))

        # + Simple linear transformation and non-linear activation
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.s1 = nn.ReLU()
        # =========================================

    def forward(self, x):
        # ============ YOUR CODE HERE =============
        x = self.linear(x)
        x = torch.matmul(self.adj_norm, x)
        x = self.s1(x)
        # =========================================
        return x

# Lets see the GCNLayer in action!
class SimpleGNN(nn.Module):
    """Simple GNN model using the GCNLayer implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.gcn_layer = GCNLayer(input_dim, output_dim, A)

    def forward(self, x):
        x = self.gcn_layer(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        y_hat = x
        return y_hat

class SimpleGNN_w_n(nn.Module):
    """Simple GNN model using the GCNLayer implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A,n):
        super(SimpleGNN_w_n, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = torch.matrix_power(A, n)
        self.gcn_layer = GCNLayer(input_dim, output_dim, self.A)

    def forward(self, x):
        x = self.gcn_layer(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        y_hat = x
        return y_hat

def train_gnn_cora(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data

def evaluate_gnn_cora(X, y, mask, model):
    model.eval()
    y_hat = model(X)[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct/num_total)
    return accuracy
    
# Training loop
def train_eval_loop_gnn_cora(model, train_x, train_y, train_mask, 
                        valid_x, valid_y, valid_mask, 
                        test_x, test_y, test_mask
                    ):
    optimiser = optim.Adam(model.parameters(), lr=LR)
    training_stats = None
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_gnn_cora(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gnn_cora(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gnn_cora(valid_x, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc = evaluate_gnn_cora(test_x, test_y, test_mask, model)
    print(f"Our final test accuracy for the SimpleGNN is: {test_acc:.3f}")
    return training_stats


