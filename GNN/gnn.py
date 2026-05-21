import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

class GNN(nn.Module):

    def __init__(self,model, hidden_dim = 64, output_dim = 2):
        super().__init__()
        self.conv1 = torch_geometric.nn.HeteroConv({
            ('metabolite', 'to', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction', 'to', 'metabolite'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('gene', 'encodes', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction','encoded_by', 'gene'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim)
        })
        self.conv2 = torch_geometric.nn.HeteroConv({
            ('metabolite', 'to', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction', 'to', 'metabolite'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('gene', 'encodes', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction','encoded_by', 'gene'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim)

        })
        self.conv3 = torch_geometric.nn.HeteroConv({
            ('metabolite', 'to', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction', 'to', 'metabolite'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('gene', 'encodes', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction','encoded_by', 'gene'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim)
        })
        self.conv4 = torch_geometric.nn.HeteroConv({
            ('metabolite', 'to', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction', 'to', 'metabolite'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('gene', 'encodes', 'reaction'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim),
            ('reaction','encoded_by', 'gene'): torch_geometric.nn.SAGEConv((-1,-1), hidden_dim)
        })
        self.num_genes = len(model.genes)
        self.num_reactions = len(model.reactions)
        self.num_metabolites = len(model.metabolites)
        self.flux_pred = nn.Sequential(
            nn.Linear(64 * (self.num_metabolites + self.num_reactions + self.num_genes), hidden_dim*2), #19456
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.corr_pred = nn.Sequential(
            nn.Linear(self.num_genes + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.input_proj = nn.ModuleDict({
            'gene': nn.Linear(1, hidden_dim),
            'reaction': nn.Linear(2, hidden_dim),
            'metabolite': nn.Linear(1, hidden_dim)
        })

    def residual(self, conv, x_dict, edge_index_dict):
        new_x_dict = conv(x_dict, edge_index_dict)
        for key in x_dict:
            if key in new_x_dict:
                new_x_dict[key] = new_x_dict[key] + x_dict[key]
        x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}
        return x_dict
    
    def forward(self, data,gene_mask, biomass_idx, target_idx):
        x_dict = data.x_dict
        x_dict = {k: self.input_proj[k](v) for k, v in x_dict.items()}
        '''x_dict = {k: self.input_proj[k](v) for k, v in x_dict.items()}
        new_x_dict = self.conv1(x_dict, data.edge_index_dict)
        for key in x_dict:
            if key in new_x_dict:
                new_x_dict[key] = new_x_dict[key] + x_dict[key]
        x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}
        new_x_dict = self.conv2(x_dict, data.edge_index_dict)
        for key in x_dict:
            if key in new_x_dict:
                new_x_dict[key] = new_x_dict[key] + x_dict[key]
        x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}
        new_x_dict = self.conv3(x_dict, data.edge_index_dict)
        for key in x_dict:
            if key in new_x_dict:
                new_x_dict[key] = new_x_dict[key] + x_dict[key]
        x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}
        new_x_dict = self.conv4(x_dict, data.edge_index_dict)
        for key in x_dict:
            if key in new_x_dict:
                new_x_dict[key] = new_x_dict[key] + x_dict[key]
        x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}'''

        x_dict = self.residual(self.conv1, x_dict, data.edge_index_dict)
        x_dict = self.residual(self.conv2, x_dict, data.edge_index_dict)
        x_dict = self.residual(self.conv3, x_dict, data.edge_index_dict)
        x_dict = self.residual(self.conv4, x_dict, data.edge_index_dict)

        graph_embedding = torch.cat([x_dict['gene'].flatten(), x_dict['reaction'].flatten(), x_dict['metabolite'].flatten()], dim=0)
        pred = self.flux_pred(graph_embedding)
        '''gene_mask = torch.tensor(gene_mask, dtype=torch.float)
        corr_input = torch.cat([gene_mask, base_pred.detach()], dim=0)
        final_pred = base_pred +  self.corr_pred(corr_input)'''

        return pred, graph_embedding


