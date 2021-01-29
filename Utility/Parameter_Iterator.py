import numpy as np
import torch
import copy

class Parameter_Iterator():
    def __init__(self, model, fixed_layers):
        self.model = model
        self.fixed_layers = fixed_layers

    def iteratate_all_parameters_and_append(self,append_list, append_parameters = False, append_zeros = False, append_bool = False,  numpy = False, to_copy = False):
        for n, m in self.model.named_modules():
         if isinstance(m, self.fixed_layers):
             for n,p in m.named_parameters():
                 if append_parameters and to_copy:
                     append_list.append(copy.deepcopy(p.data))

                 elif append_parameters and numpy:
                     append_list.append(p.data.detach().numpy())

                 elif append_zeros and not numpy:
                    zeros = torch.zeros_like(p.data)
                    append_list.append(zeros)

                 elif append_bool:
                    zeros = torch.zeros_like(p.data)
                    append_list.append(zeros > 0)
                 else:
                    raise Exception("Append Option Not Avaliable")
        return append_list

    def iteratate_all_parameters_and_apply(self, function):
        for n, m in self.model.named_modules():
         if isinstance(m, self.fixed_layers):
             for n,p in m.named_parameters():
                 function(n,p)

    def iteratate_all_parameters_apply_and_join(self, function, append_function, array):
        for n, m in self.model.named_modules():
         if isinstance(m, self.fixed_layers):
             for n,p in m.named_parameters():
                 if array is None:
                     array = function(n,p)
                 else:
                     if append_function is torch.cat or append_function is np.concatenate:
                         array = append_function([array, function(n, p)])
                     else:
                         array = append_function(array, function(n, p))
        return array
