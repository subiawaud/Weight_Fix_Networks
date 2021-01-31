import copy
import torch
import numpy as np

class Converter():
    def __init__(self, cluster_bit_fix, distance_type, zero_distance):
        self.pow_2_level = 0
        self.cluster_bit_fix = cluster_bit_fix
        self.maximum_pow_2_level = 8
        self.distance_type = distance_type
        self.zero_distance = zero_distance


    def increase_pow_2_level(self):
        print('INCREASING POW LEVEL')
        if self.pow_2_level < self.maximum_pow_2_level:
            self.pow_2_level += 1

    def convert_to_pows_of_2(self, weights, first = False):
         import math
         c = copy.deepcopy(weights.detach())
         if first:
             c[torch.abs(c) < self.zero_distance] = 0 # set small values to be zero
         c[c > 0] = torch.pow(2, torch.max(torch.Tensor([-7]).to('cuda') , torch.round(torch.log2(c[c>0]))))
         a = torch.log2(torch.abs(c[c < 0]).to('cuda'))
         c[c < 0] = -torch.pow(2, torch.max(torch.Tensor([-7]).to('cuda'), torch.round(a)))
         return c


    def convert_to_add_pows_of_2(self, weights, distance):
        current = self.convert_to_pows_of_2(weights, True)
        for x in range(self.pow_2_level):
            diff = weights - current
            next = torch.zeros(len(weights)).to('cuda')
            diff_pow_2 = self.convert_to_pows_of_2(diff)
            if self.distance_type == "relative":
                diff_dist = torch.abs(distance *  weights)
            else:
                diff_dist = distance
            to_change = torch.abs(diff) > diff_dist
            to_change = torch.logical_and(to_change, current !=0)
            next[to_change] = diff_pow_2[to_change]
            current = next + current
        return current

    def round_to_precision(self, weights, distance):
        if self.cluster_bit_fix == '32':
           return weights
        elif self.cluster_bit_fix == '16':
           return np.float16(weights)
        elif self.cluster_bit_fix == 'pow_2':
           return self.convert_to_pows_of_2(weights)
        elif self.cluster_bit_fix == 'pow_2_add':
           return self.convert_to_add_pows_of_2(weights, distance)
        else:
           raise ValueError
