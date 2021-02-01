import torch

class Distance_Calculation():
    def __init__(self, distance_type):
        self.distance_calc = self.which_distance(distance_type)

    def which_distance(self, distance_type):
        return {
           'euclidian' : self.manhatten_distance,
           'relative' : self.manhatten_distance # for now do nothing diff
        }[distance_type]


    def manhatten_distance(self, weights, centers, distance_matrix, requires_grad = False):
        for i in range(centers.size()[1]):
            if not requires_grad:
                distance_matrix[:, i] = torch.abs(weights - centers[0, i]).detach()
            else:
                distance_matrix[:, i] = torch.abs(weights - centers[0, i])

#            distance = torch.abs(weights - centers[0, i])
#            if not requires_grad:
#                distance = distance.detach().cpu()
#            if distances is None:
#                distances = distance.unsqueeze(dim = 1)
#            else:
#                try:
#                    distances = torch.stack([distances, distance], dim = 1)
#                except:
#                    distance = distance.unsqueeze(dim = 1)
#                    distances = torch.cat([distances, distance], dim = 1)
        return distance_matrix
