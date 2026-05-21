import cobra 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class MetabolicEnv_sequential(gym.Env):
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
        self.current_knockouts = set()
        with torch.no_grad():
            pred, embedding = self.gnn(self.graph, zero_mask, biomass_id, target_id)
            embedding = embedding.detach().numpy()
        self.emb_dim = embedding.shape[0]
        

        self.action_space = spaces.Discrete(self.num_genes + 1) # +1 for STOP action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.emb_dim,))

        solution = self.model.optimize()
        self.biomass_baseline = solution.fluxes[self.biomass_rxn.id]
        self.target_baseline = solution.fluxes[self.target_rxn.id]

        self.cache = {}
        self.current_knockouts = set()
        self.max_knockouts = 5

        self.curr_embedding = embedding
        self.curr_reward = 0
        self.curr_done = False
        self.curr_trucated = False
        self.curr_info = {}
        self.mask = np.zeros(self.num_genes)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        zero_mask = np.zeros(self.num_genes)
        with torch.no_grad():
            _, embedding = self.gnn(self.graph, zero_mask, self.biomass_rxn.id, self.target_rxn.id)
            embedding = embedding.detach().numpy()
        self.model = self.base_model.copy()
        self.current_knockouts = set()
        self.curr_embedding = embedding
        self.curr_reward = 0
        self.curr_done = False
        self.curr_trucated = False
        self.curr_info = {}
        self.mask = np.zeros(self.num_genes)
        return embedding, {}
    
    def step(self, action):
        done = False

        if action == self.num_genes:  # STOP action
            if np.sum(self.mask) == 0:
                self.curr_reward = -100
            self.curr_done = True
            return self.curr_embedding, self.curr_reward, self.curr_done, self.curr_trucated, self.curr_info
         
        gene = self.model.genes[action]

        if gene.id in self.current_knockouts:
            reward = -100
            done = False
            return self.curr_embedding, reward, done, False, {}
        
        self.mask[action] = 1
        self.model.genes[action].knock_out()
        self.current_knockouts.add(gene.id)
        solution = self.model.optimize()
        with torch.no_grad():
            _, embedding = self.gnn(self.graph, self.mask, self.biomass_rxn.id, self.target_rxn.id)
            embedding = embedding.detach().numpy()
        if solution.status != 'optimal':
            reward = -100
            done = True
            return embedding, reward, done, False, {}
        
        biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
        reward = 5 * (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * len(self.current_knockouts)
        if biomass_flux < 0.05:
            reward -= 5

        if len(self.current_knockouts) >= self.max_knockouts:
            done = True
        
        self.curr_done = done
        self.curr_embedding = embedding
        self.curr_reward = reward
        self.curr_trucated = False
        self.curr_info = {}

        return self.curr_embedding, self.curr_reward, self.curr_done, self.curr_trucated, self.curr_info
    

        