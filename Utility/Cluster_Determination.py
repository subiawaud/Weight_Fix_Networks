from Utility.Distance_Calculation import Distance_Calculation
from Utility.Flattener import Flattener
import torch.nn.functional as F
import numpy as np
import torch

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

#    def find_closest_centroids(self, values, number_of_clusters):
#        val, count = np.unique(values.cpu(), return_counts = True)
#        print(val, count)
#        number_of_clusters+= 1
#        if number_of_clusters > len(count):
#             idx = np.argpartition(count, -len(count))[-len(count):]
#        else:
#             idx = np.argpartition(count, -number_of_clusters)[-number_of_clusters:]
#        selected = val[idx]
#        print('selected', selected)
#        clusters = torch.Tensor([selected]).to(self.model.device)
#        e = 1e-16
#        if len(np.unique(selected)) < 2:
#            selected = np.unique(selected)
#        else:
#            selected = np.unique(selected[1:]) # we take all but the last to be our clusters
#
#        return torch.Tensor([selected]), clusters

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
        flatten_is_fixed = self.flattener.flatten_standard(self.is_fixed).detach()
        return flattened_model_weights[~flatten_is_fixed]

    def get_cluster_distances(self, weights_to_cluster = None, cluster_centers = None, only_not_fixed = True, requires_grad = False):
        if weights_to_cluster is None and only_not_fixed:
            weights_to_cluster = self.grab_only_those_not_fixed()
        elif weights_to_cluster is None:
            weights_to_cluster = self.flattener.flatten_network_tensor()
        distances = torch.zeros(weights_to_cluster.size()[0], cluster_centers.size()[1]).type_as(weights_to_cluster)
        distances = self.distance_calculator.distance_calc(weights_to_cluster, cluster_centers, distances, requires_grad)
        return distances, weights_to_cluster


    def get_cluster_assignment_prob(self, cluster_centers, requires_grad = False):
        distances, weights_to_cluster = self.get_cluster_distances(cluster_centers = cluster_centers,requires_grad =  requires_grad)
        e = 1e-12
        cluster_weight_assignment = F.softmin(distances + e, dim =1)
        weighted = self.weighting_function(cluster_weight_assignment, distances, weights_to_cluster)
        return torch.mean(weighted)
