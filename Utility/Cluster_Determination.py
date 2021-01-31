from Utility.Distance_Calculation import Distance_Calculation
from Utility.Flattener import Flattener
import torch.nn.functional as F
import numpy as np
import torch

class Cluster_Determination():
    def __init__(self, distance_calculator, model, is_fixed, distance_type, layer_shapes, flattener, zero_distance):
        self.distance_calculator = distance_calculator
        self.model = model
        self.flattener = flattener
        self.is_fixed = is_fixed
        self.zero_distance = zero_distance
        self.weighting_function = self.determine_weighting(distance_type)
        self.distance_type = distance_type
        self.layer_shapes = layer_shapes


    def determine_weighting(self, distance_type):
        return {
          'euclidian' : self.standard_weighting,
          'relative' : self.relative_weighting
        }[distance_type]

    def closest_cluster(self,weights,  clusters, iteration):
        distances, _ = self.get_cluster_distances(cluster_centers = clusters, only_not_fixed = False)
        closest_cluster, closest_cluster_index =  torch.min(distances, dim=1)
        if self.distance_type == 'relative':
            large = torch.abs(closest_cluster) > self.zero_distance
            closest_cluster[large] = torch.abs(closest_cluster[large] / (weights[large] + 1e-10))
        return closest_cluster, closest_cluster_index

    def find_closest_centroids(self, values, number_of_clusters):
        val, count = np.unique(values.cpu(), return_counts = True)
        print(val, count)
        number_of_clusters+= 1
        idx = np.argpartition(count, -number_of_clusters)[-number_of_clusters:]
        selected = val[idx]
        print('selected', selected)
        clusters = torch.Tensor([selected]).to(self.model.device)
        e = 1e-16
        if len(np.unique(selected)) < 2:
            selected = np.unique(selected)
        else:
            selected = np.unique(selected[1:]) # we take all but the last to be our clusters

        return torch.Tensor([selected]), clusters

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

    def get_cluster_distances(self, weights_to_cluster = None, cluster_centers = None, only_not_fixed = True):
        if weights_to_cluster is None and only_not_fixed:
            weights_to_cluster = self.grab_only_those_not_fixed()
        elif weights_to_cluster is None:
            weights_to_cluster = self.flattener.flatten_network_tensor()
        distances = self.distance_calculator.distance_calc(weights_to_cluster, cluster_centers)
        return distances, weights_to_cluster



    def get_cluster_assignment_prob(self, cluster_centers):
        distances, weights_to_cluster = self.get_cluster_distances(cluster_centers = cluster_centers)
        e = 1e-12
        cluster_weight_assignment = F.softmin(distances + e)
        weighted = self.weighting_function(cluster_weight_assignment, distances, weights_to_cluster)
        return torch.mean(weighted)
