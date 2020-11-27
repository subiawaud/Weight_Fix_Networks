import math
import seaborn as sns
import copy
import torch
import abc
import numpy as np
import torch.nn as nn
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.utils.prune as prune
from pytorch_lightning.metrics import functional as FM
import matplotlib.pyplot as plt
import kmeans1d
from scipy.stats import binned_statistic


LAYERS_PRUNED = (nn.Conv1d, nn.Conv2d, nn.Conv3d, P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M, nn.Linear)

class Weight_Fix_Base(pl.LightningModule):
    __metaclass_= abc.ABCMeta
    def __init__(self):
        super(Weight_Fix_Base, self).__init__()
        self.reference_image = None
        self.fixing_iteration = 0
        self.name = 'Base'
        self.optim = None
        self.outer_logger = None
        self.optim = None
        self.scheduler = None
        self.percentage_fixed = 0
        self.cluster_bit_fix = 32
        self.hist_epoch_log = 10
        self.distance_change = 0.01
        self.end_distance = 0.01
        self.clusters = 1
        self.iterations = 10
        self.is_fixed = []
        self.t = 1
        self.encourage_plus_one_cluster = True
        self.gamma = 0.25

    def set_clusters(self, iterations, number_of_clusters):
        self.clusters = 1
        self.cluster_increase = math.floor(number_of_clusters / iterations)


    def set_up(self, number_cluster_bits, end_distance, distance_change, iterations, t, gamma, encourage_plus_one_cluster):
        self.set_layer_shapes()
        self.set_inital_weights()
        self.set_up_fixed_weight_array()
        self.set_optim()
        self.encourage_plus_one_cluster = encourage_plus_one_cluster
        self.cluster_bit_fix = number_cluster_bits
        self.end_distance = end_distance
        self.distance_change = distance_change
        self.iterations = iterations
        self.fixed_center_list = torch.Tensor([[0]]).to(self.device)
        self.t = t
        self.gamma = gamma
        self.get_cluster_assignment_prob()

    def set_up_fixed_weight_array(self):
        self.fixed_weights = []
        self.is_fixed = []
        for n,p in self.named_parameters():
            zeros = np.zeros_like(p.data.detach().numpy())
            self.fixed_weights.append(zeros)
            self.is_fixed.append(np.array(zeros > 0))

    def set_inital_weights(self):
        self.initial_weights = []
        for i, (n, p) in enumerate(self.named_parameters()):
            self.initial_weights.append(copy.deepcopy(p.data))

    def reset_weights(self):
        for i, (n, p) in enumerate(self.named_parameters()):
            print('percentage=',  np.sum(self.is_fixed[i]) / len(self.is_fixed[i].flatten()))
            if 'final' in n:
                p.data[np.where(self.is_fixed[i])] = torch.Tensor(self.fixed_weights[i][np.where(self.is_fixed[i])]).to(self.device)
            else:
                new_param = self.fixed_weights[i][np.where(self.is_fixed[i])]# [self.is_fixed[i]]
                p.data[np.where(self.is_fixed[i])] = torch.Tensor(new_param).to(self.device)#.flatten()
            with torch.no_grad():
                 self.state_dict()[n] = p.data

    def set_layer_shapes(self):
        self.layer_shapes = self.get_layer_shapes()

    def on_epoch_start(self):
        if self.current_epoch == 0:
            self.custom_histogram_adder(-10)

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def set_optim(self):
        return


    def configure_optimizers(self):
        schedulers = [{
            'scheduler':self.scheduler,
            'monitor':'loss',
            'interval':'epoch',
            'frequency':1
            }]
        return [self.optim], schedulers

    def get_cluster_assignment_prob(self):
        distances, weights = self.calculate_distance_from_clusters()
        e = 0.0000001
        if len(distances.size()) < 2:
            distances = distances.unsqueeze(dim = 0)
        d = torch.sum(torch.exp(distances / self.t), axis = 1)
        s = 1 - (torch.exp(distances / self.t).T / d + e).T
        return torch.mean(torch.sum(torch.square(s * distances), axis = 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.reference_image is None:
            self.reference_image = x[0]
        y_hat = self(x)
        cluster_error = self.get_cluster_assignment_prob()
        loss = F.cross_entropy(y_hat, y) + self.gamma*cluster_error
        result = pl.TrainResult(loss)
        acc = FM.accuracy(y_hat, y)
        result.log_dict({'train_acc':acc, 'train_loss':loss}, prog_bar=True, logger = False)
        return result

    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(self, self.reference_image.unsqueeze(dim = 0))
        accuracy = torch.stack([outputs['train_acc']]).mean()
        loss = torch.stack([outputs['train_loss']]).mean()
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", accuracy, self.current_epoch)
        tensorboard_logs = {'Loss':loss, 'Accuracy': accuracy}
        epoch_dict = {'loss':loss, 'acc':accuracy}
        if self.current_epoch % self.hist_epoch_log == 0:
            self.custom_histogram_adder(None)
        return epoch_dict

    def layer_sparsity(self, params):
         with torch.no_grad():
              p = params[torch.abs(params) > 0.00001]
              number_of_zeros = params.numel() - p.numel()
              tot = params.numel()
              sparsity = number_of_zeros / tot
              return number_of_zeros, tot , sparsity

    def custom_histogram_adder(self, label = None):
        if label is None:
            label = self.current_epoch
        for name, m in self.named_modules():
            if isinstance(m, LAYERS_PRUNED):
                params = m.weight
                b_param = m.bias
                p =params[torch.abs(params) > 0.00001]
                self.logger.experiment.add_histogram(str(name) + '_weights', p, label)
                if b_param is not None:
                    self.logger.experiment.add_histogram(str(name) + '_bias', b_param, label)

    def print_the_number_of_unique_params(self, count = False):
        for n, p in self.named_parameters():
            print(n)
            if count:
                print(np.unique(p.data.cpu().detach().numpy(), return_counts = count))
            else:
                print(len(np.unique(p.data.cpu().detach().numpy())))


    def get_layer_shapes(self):
            shapes = []
            for i, (n, p) in enumerate(self.named_parameters()):
                    shapes.append(p.data.detach().cpu().numpy().shape)
            return shapes

    def flatten_the_network(self, tensor=False):
        if not tensor:
            weights = []
            for n, p in self.named_parameters():
                weights.extend(p.data.detach().flatten().cpu().numpy())
            return np.array(weights)
        else:
            weights = None
            for n, p in self.named_parameters():
                if weights is None:
                    weights = torch.flatten(p)
                else:
                    weights = torch.cat([weights,torch.flatten(p)])
            return weights

    def flatten_is_fixed(self):
       return np.concatenate([i.flatten() for i in self.is_fixed])

    def calculate_distance_from_clusters(self):
        weights = self.flatten_the_network(True)
        f = self.flatten_is_fixed()
        weights = weights[~self.flatten_is_fixed()]
        distances = None
        for t in range(self.fixed_center_list.size()[1]):
            dist = torch.square(weights - self.fixed_center_list[0,t])
            if distances is None: ## need to tidy this up
                distances = dist
            else:
                try:
                    distances = torch.stack([distances, dist], dim = 1)
                except:
                    dist = dist.unsqueeze(dim = 1)
                    distances = torch.cat([distances, dist], dim = 1)
        return distances, weights

    def summarise_clusters_selected(self, centroids, distances):
        if not self.outer_logger is None:
           self.outer_logger.experiment.add_scalar("Clusters/max_distance", np.max(distances), self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Clusters/median_distance", np.median(distances), self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Clusters/std_distance", np.std(distances), self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Clusters/mean_distance", np.mean(distances), self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Clusters/number_of_clusters", self.clusters , self.fixing_iteration)


    def select_layer_wise(self, distances, distance_allowed, percentage):
        indices = []
        count = 0
        for i, s in enumerate(self.layer_shapes):
            start = count
            count += np.prod(s)
            layer_distances = distances[start:count]
            possible_choices = (layer_distances - distance_allowed) < 0
            number_fixed = int(math.ceil(percentage*(count-start)))
            if number_fixed < (count - start):
                if np.sum(possible_choices) > number_fixed:
                    indices.extend(np.random.choice(np.where(possible_choices)[0], size = number_fixed) + start)
                else:
                    indices.extend(np.argpartition(layer_distances, number_fixed)[:number_fixed] + start)
            else:
                indices.extend(list(range(start, count)))
        return np.array(indices)




    def push_to_clusters(self, weights, percentage):
        number_fixed = int(round(len(weights)*percentage, 2))
        distance_allowed = self.end_distance + (self.iterations - self.fixing_iteration)*self.distance_change
        print('The distance allowed is ', distance_allowed)
        centroids = np.expand_dims(self.get_centroids(weights, self.clusters, distance_allowed), axis = 0) # Get the n centroids based the weight distributions
        if not self.encourage_plus_one_cluster:
            self.fixed_center_list = torch.Tensor(centroids).to(self.device)
        print('Centroids optimising for ', self.fixed_center_list[0])
        distances = np.abs(weights - centroids.transpose()).transpose() # calculate the distance for all weights from these centroids
        closest_distance = np.min(distances, axis = 1) # what is the distance to the closest centroid for each weight
        if percentage != 1.0:
            idx = self.select_layer_wise(closest_distance, distance_allowed, percentage)
        else:
           idx = range(number_fixed)
        val = np.abs(np.median(closest_distance[idx]))  + np.std(closest_distance[idx])
        print('Distance value ', val)
        if val > distance_allowed:
             self.clusters += 1
             print('INCREASING CLUSTERS', val, ' ', self.clusters)
             return self.push_to_clusters(weights, percentage)
        closest = distances.argmin(axis = 1) # which index is the closest cluster
        new_distance = np.abs(weights[idx] - centroids[0, closest[idx]]) # what is the distance from my weight to
        weights = np.zeros_like(weights)
        clustered = np.zeros_like(weights)
        weights[idx] = centroids[0, closest[idx]] # set the new weight to be the closest distance centroid (for those selected by the idx partition)
        clustered[idx] = 1
        print(len(np.unique(weights[idx])))
        self.summarise_clusters_selected(centroids, closest_distance[idx])
        return weights, idx, clustered

    def cluster_prune(self, clusters):
        count = 0
        weights, idx, clustered = self.push_to_clusters(self.flatten_the_network(), self.percentage_fixed)
        fixed = 0
        for i, s in enumerate(self.layer_shapes):
            start = count
            count += np.prod(s)
            param_weights = weights[start:count]
            self.is_fixed[i] = clustered[start:count].reshape(s) > 0
            self.fixed_weights[i] = torch.Tensor(param_weights.reshape(s))
            fixed += np.sum(self.is_fixed[i])
        print('The total fixed = ', fixed)


    def on_after_backward(self):
        for i, (n, v) in enumerate(self.named_parameters()):
             v.grad.data[np.where(self.is_fixed[i])] = torch.zeros_like(v.grad.data[np.where(self.is_fixed[i])])


    def round_to_precision(self, centroids):
        if self.cluster_bit_fix == 32:
           return centroids
        elif self.cluster_bit_fix == 16:
           return np.float16(centroids)
        else:
           raise ValueError

    def get_centroids(self, weights, clusters, distance):
        bins = int((np.max(weights) - np.min(weights)) / (distance*2))
        if bins < clusters:
            bins = clusters
        e = 1e-10
        digitized = np.histogram(weights, bins, weights = weights, density=False)[0]

        clusters += 1 # we take an extra cluster centroid but this is just for the loss term
        if bins > clusters:
            idx = np.argpartition(np.abs(digitized), -clusters)[-(clusters):]
        else:
            idx = range(len(weights))
        digitized = digitized / (np.histogram(weights, bins)[0] + e)
        selected = digitized[idx[1:]] # we take all by the last to be our clusters
        if self.encourage_plus_one_cluster:
            self.fixed_center_list = torch.Tensor([digitized[idx]]).to(self.device)
        print('Centroids selected ', selected)
        return selected

    def histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        acc = FM.accuracy(y_hat, y)
        result.log_dict({'val_acc':acc, 'val_loss':loss}, prog_bar = True, logger = True)
        return result

    def validation_epoch_end(self, outputs):
        accuracy = torch.stack([outputs['val_acc']]).mean()
        loss = torch.stack([outputs['val_loss']]).mean()
        self.logger.experiment.add_scalar("Loss/Validation", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", accuracy, self.current_epoch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True, on_epoch=True,  logger=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        acc = FM.accuracy(y_hat, y)
        y_hat = y_hat.argmax(dim=1).detach().cpu()
        y = y.detach().cpu()
        result.log_dict({'test_acc':acc, 'test_loss':loss, 'preds':y_hat, 'actual':y}, prog_bar=True, logger=False)
        return result



    def test_epoch_end(self, outputs):
        accuracy = torch.stack([outputs['test_acc']]).mean()
        loss = torch.stack([outputs['test_loss']]).mean()
        if not self.outer_logger is None:
           self.outer_logger.experiment.add_scalar("Loss/Test", loss, self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Accuracy/Test", accuracy, self.fixing_iteration)
           self.outer_logger.experiment.add_scalar("Fixed_Percentage", self.percentage_fixed, self.fixing_iteration)

        result = pl.EvalResult()
        result.log_dict({'test_acc':accuracy, 'test_loss':loss}, prog_bar=True, logger=False)
        self.custom_histogram_adder(None)
        return result
