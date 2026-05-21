import cobra 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class MetabolicEnv_gnn_embeds(gym.Env):
    def __init__(self, gnn, base_graph, biomass_id, target_id, model = 'textbook'):
        super().__init__()
        self.base_model = cobra.io.load_model(model)
        self.model = self.base_model.copy()
        self.gnn = gnn.eval()
        self.graph = base_graph
        self.target_rxn = self.model.reactions.get_by_id(target_id)
        self.biomass_rxn = self.model.reactions.get_by_id(biomass_id)
        self.num_reactions = len(self.model.reactions)
        self.num_genes = len(self.model.genes)
        zero_mask = np.zeros(self.num_genes)
        with torch.no_grad():
            pred, embedding = self.gnn(self.graph, zero_mask, biomass_id, target_id)
            embedding = embedding.detach().numpy()
        self.emb_dim = embedding.shape[0]
        

        self.action_space = spaces.MultiBinary(self.num_genes) #spaces.Discrete(self.num_reactions + 1) # +1 for STOP action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.emb_dim,))

        solution = self.model.optimize()
        self.biomass_baseline = solution.fluxes[self.biomass_rxn.id]
        self.target_baseline = solution.fluxes[self.target_rxn.id]

        self.cache = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        zero_mask = np.zeros(self.num_genes)
        with torch.no_grad():
            _, embedding = self.gnn(self.graph, zero_mask, self.biomass_rxn.id, self.target_rxn.id)
            embedding = embedding.detach().numpy()
        self.model = self.base_model.copy()
        return embedding, {}
    
    def step(self, action):
        '''done = False

        if action == self.num_reactions:  # STOP action
            done = True
            solution = self.model.optimize()
            biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
            reward = (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * len(self.current_knockouts)
            return self.get_observation(), reward, done, False, {}
        
        rxn = self.reactions[action]

        if rxn.id in self.current_knockouts:
            reward = -100
            done = False
            return self.get_observation(), reward, done, False, {}
        
        self.model.reactions.get_by_id(rxn.id).lower_bound = 0
        self.model.reactions.get_by_id(rxn.id).upper_bound = 0
        #rxn.lower_bound = 0
        #rxn.upper_bound = 0
        self.current_knockouts.add(rxn.id)
        solution = self.model.optimize()
        if solution.status != 'optimal':
            reward = -100
            done = True
            return self.get_observation(), reward, done, False, {}
        biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
        reward =  (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * len(self.current_knockouts)
        if biomass_flux < 0.05:
            reward -= 5

        if len(self.current_knockouts) >= self.max_knockouts:
            done = True
        
        observation = self.get_observation()
        info = {
            'Knockouts': self.current_knockouts,
            'Biomass Flux': biomass_flux,
            'Target Flux': target_flux
        }
        return observation, reward, done, False, info'''
        action = np.array(action)
        if tuple(action) in self.cache:
            return self.cache[tuple(action)]
        knockout_mask = action.astype(np.float32)
        with torch.no_grad():
            _, embedding = self.gnn(self.graph, knockout_mask, self.biomass_rxn.id, self.target_rxn.id)
            embedding = embedding.detach().numpy()
        for i, val in enumerate(action):
            if val == 1:
                gene = self.model.genes[i]
                gene.knock_out()
        solution = self.model.optimize()
        if solution.status != 'optimal':
            reward = -100
            done = True
            self.cache[tuple(action)] = embedding, reward, done, False, {}
            return embedding, reward, done, False, {}
        target_flux, biomass_flux = solution.fluxes[self.target_rxn.id], solution.fluxes[self.biomass_rxn.id]
        reward =  5 * (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * np.sum(action)
        if biomass_flux < 0.05:
            reward -= 5

        self.cache[tuple(action)] = embedding, reward, True, False, {}
        return embedding, reward, True, False, {}
        '''with self.model as model:
            for i, val in enumerate(action):
                if val == 1:
                    gene = self.model.genes[i]
                    gene.knock_out()
                    self.current_knockouts.add(gene.id)
            solution = model.optimize()
            if solution.status != 'optimal':
                reward = -100
                done = True
                return self.get_observation(solution), reward, done, False, {}
            biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
            reward =  (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * len(self.current_knockouts)
            if biomass_flux < 0.05:
                reward -= 5

        observation = self.get_observation(solution)
        return observation, reward, True, False, {}'''

    
    '''def get_observation(self, solution=None):
        if solution is None:
            solution = self.model.optimize()
        fluxes = solution.fluxes.values

        return fluxes
    

    def get_result(self, knockouts):
        with self.model as model:
            for gene_id in knockouts:
                gene = model.genes.get_by_id(gene_id)
                gene.knock_out()
            solution = model.optimize()
            if solution.status != 'optimal':
                return None, None
        biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
        return biomass_flux, target_flux'''