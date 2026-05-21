import cobra 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class MetabolicEnv_no_gnn(gym.Env):
    def __init__(self, biomass_id, target_id, model = 'textbook'):
        super().__init__()
        self.base_model = cobra.io.load_model(model)
        self.model = self.base_model.copy()
        self.target_rxn = self.model.reactions.get_by_id(target_id)
        self.biomass_rxn = self.model.reactions.get_by_id(biomass_id)
        self.num_reactions = len(self.model.reactions)
        self.num_genes = len(self.model.genes)
        

        self.action_space = spaces.MultiBinary(self.num_genes) #spaces.Discrete(self.num_reactions + 1) # +1 for STOP action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_reactions,))

        solution = self.model.optimize()
        self.fluxes = solution.fluxes
        self.biomass_baseline = solution.fluxes[self.biomass_rxn.id]
        self.target_baseline = solution.fluxes[self.target_rxn.id]

        self.cache = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model = self.base_model.copy()
        return self.fluxes, {}
    
    def step(self, action):
 
        if tuple(action) in self.cache:
            return self.cache[tuple(action)]
        for i, val in enumerate(action):
            if val == 1:
                gene = self.model.genes[i]
                gene.knock_out()
        solution = self.model.optimize()
        if solution.status != 'optimal':
            reward = -100
            done = True
            self.cache[tuple(action)] = solution.fluxes, reward, done, False, {}
            return solution.fluxes, reward, done, False, {}
        biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
        reward =  5 * (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * np.sum(action)
        if biomass_flux < 0.05:
            reward -= 5

        self.cache[tuple(action)] = solution.fluxes, reward, True, False, {}

        return solution.fluxes, reward, True, False, {}

