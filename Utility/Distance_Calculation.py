import torch

class Distance_Calculation():
    def __init__(self, distance_type, zero_distance):
        self.distance_calc = self.which_distance(distance_type)
        self.zero_distance = zero_distance

    def which_distance(self, distance_type):
        return {
           'euclidean' : self.manhatten_distance,
           'relative' : self.manhatten_distance # for now do nothing diff
        }[distance_type]


    def manhatten_distance(self, weights, centers, distance_matrix, requires_grad = False):
        for i in range(centers.size()[1]):
            if not requires_grad:
                distance_matrix[:, i] = torch.abs(weights - centers[0, i]).detach()
            else:
                distance_matrix[:, i] = torch.abs(weights - centers[0, i])

        return distance_matrix
