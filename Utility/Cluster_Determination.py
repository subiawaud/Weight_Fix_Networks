from Utility.Distance_Calculation import Distance_Calculation
from Utility.Flattener import Flattener
import torch.nn.functional as F
import numpy as np
import torch
import copy

class Cluster_Determination():
    def __init__(self, distance_calculator, model, is_fixed, distance_type, layer_shapes, flattener, zero_distance, device):
        self.distance_calculator = distance_calculator
        self.model = model
        self.flattener = flattener
        self.is_fixed = is_fixed
        self.zero_distance = zero_distance
        self.weighting_function = self.determine_weighting(distance_type)
        self.distance_type = distance_type
        self.layer_shapes = layer_shapes
        self.device = device

    def convert_to_pows_of_2(self, weights, p2l, first = False):
         import math
         c = copy.deepcopy(weights.detach()).type_as(weights)
         if first:
             c[torch.abs(c) < self.zero_distance] = 0 # set small values to be zero
         c[c > 0] = torch.pow(2, torch.max(torch.Tensor([-7 - p2l]).type_as(weights), torch.round(torch.log2(c[c>0]))))
         a = torch.log2(torch.abs(c[c < 0]).type_as(weights))
         c[c < 0] = -torch.pow(2, torch.max(torch.Tensor([-7 - p2l]).type_as(weights), torch.round(a)))
         return c

    def convert_to_add_pows_of_2(self, weights, distance, p2l):
        current = self.convert_to_pows_of_2(weights, p2l, True)
        for x in range(p2l):
            diff = weights - current
            next = torch.zeros(len(weights)).type_as(weights)
            diff_pow_2 = self.convert_to_pows_of_2(diff, p2l)
            diff_dist = torch.abs(distance *  weights)
            to_change = torch.abs(diff) > diff_dist
            to_change = torch.logical_and(to_change, current !=0)
            next[to_change] = diff_pow_2[to_change]
            current = next + current
        return current


    def determine_weighting(self, distance_type):
        return {
          'euclidian' : self.standard_weighting,
          'relative' : self.relative_weighting
        }[distance_type]

    def set_the_distances_of_already_fixed(self, flattened_weights, flatten_is_fixed, clusters, distances):
        if torch.sum(flatten_is_fixed) <= 1:
            return distances
        for c in range(clusters.size()[1]):
            new_vals = torch.where(flattened_weights[flatten_is_fixed] == clusters[0, c], torch.tensor(0.0).type_as(flattened_weights), torch.tensor(1.0).type_as(flattened_weights))
            distances[flatten_is_fixed, c] = new_vals
        return distances

    def closest_cluster(self,weights,  clusters, iteration):
        print('calling closest cluster', clusters, clusters.size())
        flattened_weights = self.flattener.flatten_network_tensor()
        flatten_is_fixed = self.flattener.flatten_standard(self.is_fixed).detach()
        distances = torch.ones(flattened_weights.size()[0], clusters.size()[1]).to('cuda') # create a zero matrix for each of the distances to clusters
        distances = self.set_the_distances_of_already_fixed(flattened_weights, flatten_is_fixed, clusters, distances)
        newly_fixed_distances = self.distance_calculator.distance_calc(flattened_weights[~flatten_is_fixed], clusters, distances[~flatten_is_fixed], requires_grad = False)
        distances[~flatten_is_fixed] = newly_fixed_distances
       
        closest_cluster, closest_cluster_index =  torch.min(distances, dim=1)
        if self.distance_type == 'relative':
            small = (torch.abs(weights) < self.zero_distance)
            print('index of small', torch.where(small))
            print('percent small', torch.sum(small))
            print(len(weights))
            closest_cluster[~small] = torch.abs(closest_cluster[~small] / (weights[~small]))
#            closest_cluster[small] = 0
        return closest_cluster, closest_cluster_index

    def get_clusters(self, mi, d):
        if mi < 0:
            ma = (d*mi + mi)*(1 - d) 
        else:
            ma = (d*mi + mi)/(1 - d)
        return ma 



    def create_possible_centroids(self, max_weight, min_weight, a, powl):
           vals = np.zeros(10000)
           ma = min_weight 
           i = 0 
           while(ma < max_weight):
                 vals[i] = ma
                 if abs(ma) < self.zero_distance:
                      ma = 0
                 if ma == 0:
                      ma = self.zero_distance
                 else:
                      ma = self.get_clusters(ma, a)
                 i += 1
           return torch.unique(self.convert_to_add_pows_of_2(torch.Tensor(vals),a, powl)) 

    def find_the_next_cluster(self, weights, is_fixed, vals, zero_index,  max_dist, to_cluster):
        distances = (torch.abs(weights[~is_fixed] - vals.unsqueeze(1) ) / (torch.abs(weights[~is_fixed])+1e-7)) # take the distances between weights and vals
        if zero_index:
            distances[zero_index,(torch.abs(weights[~is_fixed] )<self.zero_distance).squeeze()] = 0 
        u, c = torch.unique(distances.argmin(0), return_counts = True)
        am = u[torch.argmax(c)] # which cluster has the most local weights   
        local_cluster_distances = distances[am, :]
        sorted_indexes = np.argsort(local_cluster_distances) #sort them by index
#        rolling_mean = np.cumsum(local_cluster_distances[sorted_indexes]) / np.arange(1, len(sorted_indexes)+1)
        first_larger = np.argmax(local_cluster_distances[sorted_indexes]  >= max_dist)  # which is the first distance larger than the max_dist
        if first_larger > to_cluster or (first_larger == 0 and local_cluster_distances[-1] <= max_dist):
            return sorted_indexes[:to_cluster], vals[am], local_cluster_distances[sorted_indexes[:to_cluster]]
        else:
            return sorted_indexes[:first_larger], vals[am], local_cluster_distances[sorted_indexes[:first_larger]]
    
    def get_the_clusters(self, percent, a):
        weights = self.flattener.flatten_network_tensor().detach().cpu()
        #is_fixed = self.flattener.flatten_standard(self.is_fixed).detach()
        is_fixed = torch.zeros_like(weights).bool().detach()
        taken = 0
        ap2 = 0
        to_take = int(len(weights)*percent)
        clusters = []
        new_weight_set = torch.zeros_like(weights).type_as(weights)
        distances = torch.zeros_like(weights).type_as(weights)
        while(taken < to_take):
           vals = self.create_possible_centroids(torch.max(weights[~is_fixed]), torch.min(weights[~is_fixed]), a, ap2).type_as(weights)
           needed = to_take - taken
           indicies, cluster, dist  = self.find_the_next_cluster(weights, is_fixed, vals, torch.where(vals==0), max_dist=a, to_cluster = needed)
           parent_idx = np.arange(weights.size()[0])[~is_fixed.squeeze()][indicies]
           print(parent_idx, 'taking')
           print(dist)
           new_weight_set[parent_idx] = cluster
           distances[parent_idx] = dist#.unsqueeze(1)
           taken += len(indicies)
           clusters.append(cluster)
           is_fixed[parent_idx] = True
           if len(indicies) == 0:
               print('increasing')
               ap2 += 1
        clusters = torch.unique(torch.Tensor(clusters)).type_as(weights)
        self.is_fixed = ~is_fixed
        return clusters.unsqueeze(0), is_fixed, distances, new_weight_set

    def select_layer_wise(self, distances, distance_allowed, percentage):
        distances = distances.detach().cpu().numpy()
        indices = []
        count = 0
        for i, s in enumerate(self.layer_shapes):
            start = count
            count += int(np.prod(s))
            layer_distances = np.nan_to_num(distances[start:count])
            possible_choices = (layer_distances) - distance_allowed < 0
            number_fixed = int(round(percentage*(count-start)))
            if number_fixed < (count - start):
                if np.sum(possible_choices) > number_fixed:
                    indices.extend(np.argpartition(layer_distances, number_fixed)[:number_fixed] + start)
                else:
                    indices.extend(np.argpartition(layer_distances, number_fixed)[:number_fixed] + start)
            else:
                indices.extend(list(range(start, count)))
        return np.array(indices)

    def select_not_layer_wise(self, distances, distance_allowed, percentage):
        distances = distances.detach().cpu().numpy()
        number = int(len(distances)*percentage)
        smallest_idx = np.argpartition(distances, number)[:number]
        print('distances in select', distances[smallest_idx])
        print('smallest' , smallest_idx)
        return smallest_idx
        

    def standard_weighting(self, weighting, distance):
        return torch.sum(torch.square(weighting * distance), axis =1)

    def relative_weighting(self, weighting, distance, weights):
        weighted = self.standard_weighting(weighting, distance)
        larger = torch.abs(weighted) > self.zero_distance
        weighted[larger] = torch.div(weighted[larger], torch.abs(weights[larger]))
        return weighted

    def grab_only_those_not_fixed(self):
        flattened_model_weights = self.flattener.flatten_network_tensor()
        return flattened_model_weights[self.is_fixed]

    def get_cluster_distances(self, is_fixed = None, cluster_centers = None, only_not_fixed = True, requires_grad = False):
        if is_fixed is None and only_not_fixed:
            is_fixed = self.grab_only_those_not_fixed()
        elif is_fixed is None:
            is_fixed = self.flattener.flatten_network_tensor()
        distances = torch.zeros(is_fixed.size()[0], cluster_centers.size()[1]).type_as(is_fixed)
        distances = self.distance_calculator.distance_calc(is_fixed, cluster_centers, distances, requires_grad)
        return distances, is_fixed


    def get_cluster_assignment_prob(self, cluster_centers, requires_grad = False):
        distances, is_fixed = self.get_cluster_distances(cluster_centers = cluster_centers,requires_grad =  requires_grad)
        e = 1e-12
        cluster_weight_assignment = F.softmin(distances + e, dim =1)
        weighted = self.weighting_function(cluster_weight_assignment, distances, is_fixed)
        return torch.mean(weighted)
