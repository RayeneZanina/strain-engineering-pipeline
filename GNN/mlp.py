import torch
import torch.nn as nn
import numpy as np

class GeneMLP(nn.Module):
    def __init__(self, num_genes):
        super().__init__()
        self.num_genes = num_genes
        self.network = nn.Sequential(
            nn.Linear(num_genes, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, gene_vector):
        return self.network(gene_vector)

def train_mlp(m, nn,batch_size = 32, epochs = 500, lr = 0.0001):
    model = m.copy()

    genes = list(model.genes)
    reactions = list(model.reactions)
    metabolites = list(model.metabolites)

    genes_to_idx = {gene.id: idx for idx, gene in enumerate(genes)}
    rxn_to_idx = {rxn.id: idx for idx, rxn in enumerate(reactions)}
    met_to_idx = {met.id: idx for idx, met in enumerate(metabolites)}
    
    genes = [g.id for g in model.genes]
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    biomass_idx = rxn_to_idx[model.reactions.get_by_id('Biomass_Ecoli_core').id]
    target_idx = rxn_to_idx[model.reactions.get_by_id('EX_etoh_e').id]
    for epoch in range(epochs):
        losses = []
        for _ in range(batch_size):
            m = model.copy()
            knockouts = np.random.rand(nn.num_genes) < 0.05
            for idx, knockout in enumerate(knockouts):
                if knockout:
                    m.genes.get_by_id(genes[idx]).knock_out()
            solution = m.optimize()
            if solution.status != 'optimal':
                continue
            biomass_flux = np.log1p(solution.fluxes[m.reactions.get_by_id('Biomass_Ecoli_core').id])
            target_flux = np.log1p(solution.fluxes[m.reactions.get_by_id('EX_etoh_e').id])
            target = torch.tensor([target_flux , biomass_flux ], dtype=torch.float)
            gene_vector = torch.tensor([gene.functional for gene in m.genes], dtype=torch.float).unsqueeze(0)
            pred = nn(gene_vector)
            loss = torch.nn.functional.mse_loss(pred, target)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')