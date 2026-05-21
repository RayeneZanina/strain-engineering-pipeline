import numpy as np
import torch
from torch_geometric.data import HeteroData
import tqdm

def build_graph(m):
    model = m.copy()
    data = HeteroData()

    genes = list(model.genes)
    reactions = list(model.reactions)
    metabolites = list(model.metabolites)

    genes_to_idx = {gene.id: idx for idx, gene in enumerate(genes)}
    rxn_to_idx = {rxn.id: idx for idx, rxn in enumerate(reactions)}
    met_to_idx = {met.id: idx for idx, met in enumerate(metabolites)}
    
    gene_data = [1] * len(genes)
    reactions_data = [[r.lower_bound, r.upper_bound] for r in model.reactions]
    metabolites_data = [len(met.reactions) for met in model.metabolites]

    data['gene'].x = torch.tensor(gene_data, dtype=torch.float).unsqueeze(1)
    data['reaction'].x = torch.tensor(reactions_data, dtype=torch.float)
    data['metabolite'].x = torch.tensor(metabolites_data, dtype=torch.float).unsqueeze(1)

    encod_gene = []
    encod_rxn = []
    consum_rxn = []
    consum_met = []
    prod_rxn = []
    prod_met = []
    for rxn in reactions:
        rxn_idx = rxn_to_idx[rxn.id]
        for gene in rxn.genes:
            gene_idx = genes_to_idx[gene.id]
            encod_rxn.append(rxn_idx)
            encod_gene.append(gene_idx)
        for met, coeff in rxn.metabolites.items():
            met_idx = met_to_idx[met.id]
            if coeff < 0 or rxn.reversibility:
                consum_rxn.append(rxn_idx)
                consum_met.append(met_idx)
            if coeff > 0 or rxn.reversibility:
                prod_rxn.append(rxn_idx)
                prod_met.append(met_idx)

    data['metabolite', 'to', 'reaction'].edge_index = torch.tensor([consum_met, consum_rxn], dtype=torch.long)
    data['reaction', 'to', 'metabolite'].edge_index = torch.tensor([prod_rxn, prod_met], dtype=torch.long)
    data['gene', 'encodes', 'reaction'].edge_index = torch.tensor([encod_gene, encod_rxn], dtype=torch.long)
    data['reaction','encoded_by', 'gene'].edge_index = torch.tensor([encod_rxn, encod_gene], dtype=torch.long)

    return data, rxn_to_idx

def update_graph(model, data, gene_knockouts):
    data_clone = data.clone()
    m = model.copy()

    for gene in gene_knockouts:
        m.genes.get_by_id(gene).knock_out()
    new_gene_data = [gene.functional for gene in m.genes]
    new_reaction_data = [[r.lower_bound, r.upper_bound] for r in m.reactions]
    data_clone['gene'].x = torch.tensor(new_gene_data, dtype=torch.float).unsqueeze(1)
    data_clone['reaction'].x = torch.tensor(new_reaction_data, dtype=torch.float)

    return data_clone

def generate_sample(model, data, biomass_id, target_id, knockout_prob = 0.05):
    data_clone = data.clone()
    m = model.copy()
    genes = np.array([g.id for g in model.genes])
    knockouts = np.random.rand(len(genes)) < knockout_prob
    #new_graph = update_graph(model, data, gene_id_knockouts)

    for i,gene in enumerate(genes):
        if knockouts[i]:
            m.genes.get_by_id(gene).knock_out()
    new_gene_data = [gene.functional for gene in m.genes]
    new_reaction_data = [[r.lower_bound, r.upper_bound] for r in m.reactions]
    data_clone['gene'].x = torch.tensor(new_gene_data, dtype=torch.float).unsqueeze(1)
    data_clone['reaction'].x = torch.tensor(new_reaction_data, dtype=torch.float)
    solution = m.optimize()
    if solution.status != 'optimal':
        return None, None, knockouts
    biomass_flux = solution.fluxes[biomass_id]
    target_flux = solution.fluxes[target_id]
    
    target = torch.tensor([np.log1p(target_flux) , np.log1p(biomass_flux) ], dtype=torch.float)
    return data_clone, target, knockouts

def train(m, gnn, biomass_id, target_id, batch_size = 24, epochs = 300, lr = 0.0001):
    model = m.copy()
    base_graph, rxn_to_idx = build_graph(model)
    biomass_idx = rxn_to_idx[biomass_id]
    target_idx = rxn_to_idx[target_id]
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = []
        for _ in range(batch_size):
            graph, target, knockouts = generate_sample(model, base_graph, biomass_id, target_id)
            if target is None:
                continue
            pred, embedding = gnn(graph,knockouts,biomass_idx,target_idx )
            total_loss.append(torch.nn.functional.mse_loss(pred, target))
        loss = torch.mean(torch.stack(total_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
