import torch
import numpy as np

class Flattener():
    def __init__(self, parameter_iterator, fixed_layers):

        self.parameter_iterator=parameter_iterator
        self.fixed_layers = fixed_layers

    def flatten_standard(self, data):
       try:
           return torch.cat([i.flatten() for i in data])
       except:
           return np.concatenate([i.flatten() for i in data])

    def flatten_network_tensor(self):
            return self.parameter_iterator.iteratate_all_parameters_apply_and_join(self.flatten, torch.cat, None)

    def flatten_numpy(self, n, p):
                    return p.data.detach().flatten().cpu().numpy()

    def flatten(self, n,  p):
                    return torch.flatten(p)

    def flatten_network_numpy(self):
            return self.parameter_iterator.iteratate_all_parameters_apply_and_join(self.flatten_numpy, np.concatenate, None)
