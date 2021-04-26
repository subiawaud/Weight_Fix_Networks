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
        self.is_fixed_flat = self.flattener.flatten_standard(is_fixed).detach()
        self.zero_distance = zero_distance
        self.weighting_function = self.determine_weighting(distance_type)
        self.distance_type = distance_type
        self.layer_shapes = layer_shapes
        self.device = device
        self.not_fixed = torch.where(~self.is_fixed_flat)[0].to(self.device)


    def convert_to_pows_of_2(self, weights, p2l, first = False):
         import math
         c = copy.deepcopy(weights.detach()).type_as(weights)
         if first:
             c[torch.abs(c) < self.zero_distance] = 0 # set small values to be zero
         c[c > 0] = torch.pow(2, torch.max(torch.Tensor([np.log2(self.zero_distance) - p2l - 1]).type_as(weights), torch.round(torch.log2(c[c>0]))))
         a = torch.log2(torch.abs(c[c < 0]).type_as(weights))
         c[c < 0] = -torch.pow(2, torch.max(torch.Tensor([np.log2(self.zero_distance) - p2l - 1]).type_as(weights), torch.round(a)))
         return c

    def convert_to_add_pows_of_2(self, weights, distance, p2l):
        current = self.convert_to_pows_of_2(weights, p2l, True)
        for x in range(p2l):
            diff = weights - current
            next = torch.zeros(len(weights)).type_as(weights)
            diff_pow_2 = self.convert_to_pows_of_2(diff, p2l)
            diff_dist = torch.abs(distance *  weights)
            to_change = torch.abs(diff) >= diff_dist
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
        flatten_is_fixed = self.is_fixed_flat
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
        ma = (mi)/(1 - d) # if we use the mean dist
         #   ma = mi/(1-d) # max dist use
        return ma 



    def create_possible_centroids(self, max_weight, min_weight, a, powl):
           vals = np.zeros(100000)
           boundary = max(abs(min_weight), max_weight)
           ma = self.zero_distance
           vals[0] = 0
           i = 0
           while(ma < boundary):
                 vals[i] = ma
                 ma = self.get_clusters(ma, a)
                 i += 1
           print('vals before', vals[:40])
           pw = torch.unique(self.convert_to_add_pows_of_2(torch.Tensor(vals),a, powl))
           print('vals after', pw[:40])
           return torch.unique(torch.cat([torch.flip(-pw, [0]), pw]))


    def find_the_next_cluster(self, weights, is_fixed, vals, zero_index,  max_dist, to_cluster):
        #fixed_but_not_fixed = torch.logical_and(already_fixed, ~is_fixed)
        print('weight values', weights)

        distances = torch.abs(weights[~is_fixed] - vals.unsqueeze(1))  # take the distances between weights and vals
        #if there is still an issue, it will be related to the assignment of closest and not that distance between (me thinks)
        print('vals at start', vals)
        print('distances at the start', distances[:,0])

        print('min dists', distances.min(0))
        print('min selected  =', distances.argmin(0))
        u, c = torch.unique(distances.argmin(0), return_counts = True)
        print('uc', u,c)
        print('argmax c', torch.argmax(c))
        am = u[torch.argmax(c)] # which cluster has the most local weights


        distances /=  (torch.abs(weights[~is_fixed]))
        print('dbz1', distances[distances != distances])
        distances[distances != distances] = 1 # here we overcome the divide by zero issue
        print('dbz2', distances[distances != distances])
        if zero_index:
            distances[zero_index,(torch.abs(weights[~is_fixed]) < self.zero_distance).squeeze()] = 0
        print('dist of zero index', distances[zero_index])
        print('dist after normalisation', distances[:,0])

        print('cluster selected =', am, vals[am])
        local_cluster_distances = distances[am, :]
        print('local cluster distances =', local_cluster_distances)
        sorted_indexes = np.argsort(local_cluster_distances) #sort them by index
        print('weights at local dists', weights[~is_fixed][sorted_indexes])
        print('sorted local cluster distances', local_cluster_distances[sorted_indexes])
        if torch.sum((local_cluster_distances[sorted_indexes] <= 0.00)) > 1:
            first_larger_than_zero = min(np.max(np.where(local_cluster_distances[sorted_indexes]  <=  0.00)[0])+1, len(sorted_indexes))  # which is the first distance > 0
        else:
            print('none <= 0')
            first_larger_than_zero = 0
        print('fltz', first_larger_than_zero)

        #zero will be from previous  assignments usually )

        divide_by = np.ones(len(sorted_indexes))
        divide_by[first_larger_than_zero:] = np.arange(1, len(sorted_indexes) - first_larger_than_zero + 1)

        rolling_mean = np.cumsum(local_cluster_distances[sorted_indexes]) / divide_by
        print('rm', rolling_mean)
        try:
            print('rolling < max dist', np.where(rolling_mean <= max_dist))
            print('rolling < max dist 2', np.where(rolling_mean <= max_dist)[0])
            first_larger = np.max(np.where(rolling_mean <= max_dist)[0]) + 1
        except Exception as e:
            print('in except', e)
            first_larger = 0


        if first_larger > to_cluster or (first_larger == 0 and local_cluster_distances[-1] <= max_dist):
            print(1, first_larger > to_cluster)
            print(2, (first_larger == 0 and local_cluster_distances[-1] <= max_dist))
            return sorted_indexes[:to_cluster], vals[am], local_cluster_distances[sorted_indexes[:to_cluster]]
        else:
            print(3)
            return sorted_indexes[:first_larger], vals[am], local_cluster_distances[sorted_indexes[:first_larger]]
    
    def get_the_clusters(self, percent, dist_allowed):
        weights = self.flattener.flatten_network_tensor()
        dev = weights.device
        weights = weights.detach().cpu()
        #is_fixed = self.flattener.flatten_standard(self.is_fixed).detach()
        is_fixed = torch.zeros_like(weights).bool().detach()
        taken = 0
        ap2 = 0
        to_take = int(len(weights)*percent)
        clusters = []
        distances = torch.zeros_like(weights).type_as(weights)
        a = dist_allowed #we separate this in order to allow distance allowed to grow but not the cluster distribution
        while(taken < to_take):
           vals = self.create_possible_centroids(torch.max(weights[~is_fixed]), torch.min(weights[~is_fixed]), dist_allowed, ap2).type_as(weights)
           print('these are the values', vals)
           print('i am a', a)
           print('i am the distance allowed', dist_allowed)
           needed = to_take - taken
           indicies, cluster, dist  = self.find_the_next_cluster(weights, is_fixed, vals, torch.where(vals==0), a, needed)
           parent_idx = np.arange(weights.size()[0])[~is_fixed.squeeze()][indicies]
           print('the weights clustered =', weights[parent_idx])
           weights[parent_idx] = cluster
           distances[parent_idx] = dist#.unsqueeze(1)
           taken += len(indicies)
           clusters.append(cluster)
           is_fixed[parent_idx] = True
           print('taking', len(indicies))
           if len(indicies) == 0:
               print('increasing ap2', ap2)
               ap2 += 1
               if ap2 > 10:
                   a *= 2
           else:
                a = dist_allowed
        clusters = torch.unique(torch.Tensor(clusters))
        self.is_fixed_flat = is_fixed.to(self.device)
        self.not_fixed = torch.where(~self.is_fixed_flat)[0].to(dev)
        return clusters.unsqueeze(0), self.is_fixed_flat, distances, weights

        

    def standard_weighting(self, weighting, distance):
        return torch.sum(torch.square(weighting * distance), axis =1)

    def relative_weighting(self, weighting, distance, weights):
        weighted = self.standard_weighting(weighting, distance)
        weighted = torch.div(weighted, torch.abs(weights))
        weighted = torch.where(torch.abs(weights) > self.zero_distance, weights, torch.zeros_like(weights, device=weighted.device))
        return weighted


    def grab_only_those_not_fixed(self):
        flattened_model_weights = self.flattener.flatten_network_tensor()
        return torch.index_select(flattened_model_weights,0, self.not_fixed)


    def get_cluster_distances(self, is_fixed = None, cluster_centers = None, only_not_fixed = True, requires_grad = False):
     #   if is_fixed is None and only_not_fixed:
        weights_not_fixed = self.grab_only_those_not_fixed()
       # elif is_fixed is None:
        #    is_fixed = self.flattener.flatten_network_tensor()
        distances = torch.zeros(weights_not_fixed.size()[0], cluster_centers.size()[1], device=weights_not_fixed.device)
        distances = self.distance_calculator.distance_calc(weights_not_fixed, cluster_centers, distances, requires_grad)
        return distances, weights_not_fixed



    def get_cluster_assignment_prob(self, cluster_centers, requires_grad = False):
        distances, is_fixed = self.get_cluster_distances(cluster_centers = cluster_centers,requires_grad =  requires_grad)
        e = 1e-12
        cluster_weight_assignment = F.softmin(distances + e, dim =1)
        weighted = self.weighting_function(cluster_weight_assignment, distances, is_fixed)
        return torch.mean(weighted)
