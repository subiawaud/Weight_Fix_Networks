import math
import seaborn as sns
import copy
import torch
import abc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.utils.prune as prune
from pytorch_lightning.metrics import Accuracy
from Models.CaptureOutputs import CaptureOutputs
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from Utility.Cluster_Determination import Cluster_Determination
from Utility.Converter import Converter
from Utility.Distance_Calculation import Distance_Calculation
from Utility.Flattener import Flattener
from Utility.Metric_Capture import Metric_Capture
from Utility.Parameter_Iterator import *
from scipy.stats import entropy
from kmeans_pytorch import kmeans



class Weight_Fix_Base(pl.LightningModule):
    __metaclass_= abc.ABCMeta # to allow for abstract method implementations
    def __init__(self):
        super(Weight_Fix_Base, self).__init__()
        self.current_fixing_iteration = 0 # which iteration of the fixing algorithm are we at
        self.name = 'Base'  # object name
        self.tracking_gradients = False
        self.percentage_fixed = 0
        self.layers_fixed = None

        self.taccuracy = pl.metrics.Accuracy()
        self.ttaccuracy = pl.metrics.Accuracy()
        self.tt5accuracy = pl.metrics.Accuracy(top_k=5)
        self.vaccuracy = pl.metrics.Accuracy()

    def reset_optim(self, max_epochs):
        self.max_epochs = max_epochs
        self.set_optim(max_epochs, self.lr)
        self.parameter_iterator = Parameter_Iterator(self, self.layers_fixed)
        self.flattener = Flattener(self.parameter_iterator, self.is_fixed)
        self.cluster_determinator = Cluster_Determination(self.distance_calculator, self, self.is_fixed, self.calculation_type, self.layer_shapes, self.flattener, self.zero_distance, self.device)
        self.metric_logger = Metric_Capture(self)

    def set_loggers(self, inner, outer):
        self.metric_logger.set_loggers(inner, outer)

    def set_up(self, distance_calculation_type, cluster_bit_fix, smallest_distance_allowed, number_of_fixing_iterations, regularisation_ratio, how_many_iterations_not_regularised, zero_distance, bn_inc):
        if bn_inc:
                print('BATCH NORM included')
                self.layers_fixed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm2d, nn.BatchNorm1d)
        else:
                self.layers_fixed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        self.zero_distance = zero_distance
        self.parameter_iterator = Parameter_Iterator(self, self.layers_fixed)
        self.set_layer_shapes()
        self.set_up_fixed_weight_array()
        self.set_inital_weights()
        self.calculation_type = distance_calculation_type
        self.distance_calculator = Distance_Calculation(self.calculation_type)
        self.flattener = Flattener(self.parameter_iterator, self.is_fixed)
        self.cluster_determinator = Cluster_Determination(self.distance_calculator, self, self.is_fixed, self.calculation_type, self.layer_shapes, self.flattener, zero_distance, self.device)
        self.metric_logger = Metric_Capture(self)

        self.converter = Converter(cluster_bit_fix, distance_calculation_type, zero_distance, self.device)
        self.smallest_distance_allowed = smallest_distance_allowed
        self.cluster_bit_fix = cluster_bit_fix
        self.number_of_fixing_iterations = number_of_fixing_iterations
        self.how_many_iterations_not_regularised = how_many_iterations_not_regularised
        self.regularisation_ratio = regularisation_ratio
        self.number_of_clusters = 3

    def set_up_fixed_weight_array(self):
        self.fixed_weights = self.parameter_iterator.iteratate_all_parameters_and_append([], append_zeros=True)
        self.is_fixed = self.parameter_iterator.iteratate_all_parameters_and_append([], append_bool = True)
        self.grads = self.parameter_iterator.iteratate_all_parameters_and_append([], append_zeros=True)

    def set_inital_weights(self):
        self.initial_weights = self.parameter_iterator.iteratate_all_parameters_and_append([], to_copy = True, append_parameters = True)

    def reset_weights(self):
        i = 0
        for n, m in self.named_modules():
         if isinstance(m, self.layers_fixed):
             for n, p in m.named_parameters():
                new_param = self.fixed_weights[i][torch.where(self.is_fixed[i])]# [self.is_fixed[i]]
                p.data[torch.where(self.is_fixed[i])] = new_param # torch.Tensor(new_param).to(self.device)#.flatten()
                i += 1
                with torch.no_grad():
                     self.state_dict()[n] = p.data

    def set_layer_shapes(self):
        self.layer_shapes = self.get_layer_shapes()

    def update_results(self, exp_name, orig_acc, orig_entropy, orig_params, test_acc, fixing_epochs, data_name, zd, bn):
        fixed_params = self.get_number_of_u_params()
        fixed_entropy, fixed_entropy_non_zero = self.get_weight_entropy()
        self.metric_logger.write_to_results_file(exp_name, self.name, self.regularisation_ratio,
        self.smallest_distance_allowed, fixing_epochs, orig_acc, orig_entropy, orig_params, test_acc, fixed_entropy, fixed_entropy_non_zero, fixed_params, data_name, zd, bn)


    def get_number_of_u_params(self):
        return len(np.unique(self.flattener.flatten_network_numpy()))

    def get_weight_entropy(self):
        v, c = np.unique(self.flattener.flatten_network_numpy(), return_counts = True)
        try:
                c_z = np.delete(c, v.index(0.))
        except:
	        c_z = c
	        print('no zero entry')
        c = np.array(c) / np.sum(c)
        c_z = np.array(c_z) / np.sum(c_z)
        return entropy(c, base=2), entropy(c_z, base=2)

    def on_epoch_start(self):
        if self.current_epoch == 0:
            print('starting')

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def set_optim(self, epochs):
        return

    def configure_optimizers(self):
        scheduler = {
            'scheduler':self.scheduler,
            'name':'learning_rate',
            'interval':'step',
            }

        if self.scheduler is not None:
            return self.optim, [scheduler]
        else:
            print('no scheduler')
            return self.optim

    def calculate_cluster_error_alpha(self, ce, cluster_error):
        if self.current_fixing_iteration > self.number_of_fixing_iterations - self.how_many_iterations_not_regularised:
            return 0
        else:
#            increase_factor = (self.number_of_fixing_iterations - self.current_fixing_iteration)
            alpha = ((self.regularisation_ratio*ce)/cluster_error).detach().clone().to(self.device)
            return alpha

    def calculate_cluster_error(self, ce):
        if self.regularisation_ratio  > 0:
            cluster_error = self.cluster_determinator.get_cluster_assignment_prob(self.centroid_to_regularise_to, requires_grad = True)
            alpha = self.calculate_cluster_error_alpha(ce, cluster_error)
            return alpha * cluster_error
        else:
            return 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        ce = F.cross_entropy(y_hat, y)
        cluster_error = self.calculate_cluster_error(ce)
        loss = cluster_error + ce
        acc = self.taccuracy(F.softmax(y_hat, dim=1), y)
        self.log('train_loss',loss, prog_bar=False, logger = False, sync_dist=True, on_epoch=True)
        return loss

    def training_epoch_end(self, o):
         self.log('train_acc_epoch', self.taccuracy.compute())

    def grab_shape(self, n, p):
        return p.shape


    def get_layer_shapes(self):
            shapes = []
            for n, m in self.named_modules():
             if isinstance(m, self.layers_fixed):
                 for n, p in m.named_parameters():
                        shapes.append(p.data.detach().cpu().numpy().shape)
            return shapes

    def flatten_grads(self):
        return self.flattener.flatten_standard(self.grads)

    def flatten_is_fixed(self):
        return self.flattener.flatten_standard(self.is_fixed)

    def determine_which_weights_are_newly_fixed(self, fixed, flattened_network):
        currently_fixed_indicies = np.argwhere(flattened_network.cpu())
        new_fixed = np.setdiff1d(np.argwhere(fixed.cpu()), currently_fixed_indicies)
#        new_fixed = np.setdiff1d(np.union1d(idx, currently_fixed_indicies), np.intersect1d(idx, currently_fixed_indicies))
        print('currently fixed indicies', currently_fixed_indicies)
        print('newly fixed indexes', new_fixed)
        return new_fixed

    def calculate_threshold_value(self, distances_of_newly_fixed):
        print('distances to be fixed', distances_of_newly_fixed)
        print('mean to be fixed', torch.mean(distances_of_newly_fixed))
        print('max to be fixed', torch.max(distances_of_newly_fixed))
        return torch.mean(distances_of_newly_fixed) + 1*torch.std(distances_of_newly_fixed)

    def calculate_allowable_distance(self):
        a = max(self.smallest_distance_allowed, self.smallest_distance_allowed*((self.number_of_fixing_iterations - self.current_fixing_iteration)))
        return a

    def calculate_how_many_to_fix(self, weights, percentage):
        return int(round(len(weights)*percentage, 2))


    def gather_assigned_clusters(self, centroids, idx, closest_cluster_list,  previous_weights):
        weights = torch.zeros_like(previous_weights).to(self.device)
        clustered = torch.zeros_like(previous_weights).to(self.device)
        weights[idx] = centroids[0, closest_cluster_list[idx]].to(self.device) # set the new weight to be the closest distance centroid (for those selected by the idx partition)
        clustered[idx] = 1
        return weights, clustered

    def assign_weights_to_clusters(self, clustered_weights, is_clustered_list):
        count = 0
        fixed = 0
        for i, s in enumerate(self.layer_shapes):
            start = count
            count += np.prod(s)
            self.is_fixed[i] = is_clustered_list[start:count].reshape(s) > 0
            print(f'Fixed this layer {i} is {torch.sum(self.is_fixed[i])}')
            self.fixed_weights[i] = clustered_weights[start:count].reshape(s).to(self.device)
            fixed += torch.sum(self.is_fixed[i])


    def print_unique_params(self):
        return self.parameter_iterator.iteratate_all_parameters_and_apply(self.metric_logger.print_the_number_of_unique_params)

    def apply_clustering_to_network(self, quantised_weights = None):
        weights = self.flattener.flatten_network_tensor()
        percentage = self.percentage_fixed
        number_fixed = self.calculate_how_many_to_fix(weights, percentage)
        centroids, is_fixed, closest_cluster_distance, clustered_weights = self.cluster_determinator.get_the_clusters(percentage, self.calculate_allowable_distance())
        print('centroids chosen', centroids)
        print('is fixed chosen', torch.sum(is_fixed))
        print(is_fixed.size(), closest_cluster_distance, clustered_weights)
        print(self.flattener.flatten_standard(self.is_fixed).size())
        self.centroid_to_regularise_to = centroids.detach()
        newly_fixed = self.determine_which_weights_are_newly_fixed(is_fixed, self.flattener.flatten_standard(self.is_fixed))
        threshold_val = self.calculate_threshold_value(closest_cluster_distance[newly_fixed])
        print('threshold val is', self.calculate_allowable_distance())
        print('current val  ', threshold_val)
        self.metric_logger.summarise_clusters_selected(centroids, closest_cluster_distance[newly_fixed],threshold_val, self.smallest_distance_allowed, self.current_fixing_iteration)
        self.assign_weights_to_clusters(clustered_weights.detach(), is_fixed.detach())
    #    self.reset_cluster_nums()

    def reset_cluster_nums(self):
        self.number_of_clusters = 3
        self.converter.reset()
        

    def save_clusters(self):
        print('not needed')
        
    def update_gradient_data_tracker(self, i, grad):
        if self.tracking_gradients:
            self.grads[i] = np.abs(grad.data.cpu().detach().numpy())

    def on_after_backward(self):
        i = 0
        for n, pp in self.named_modules():
          if isinstance(pp, self.layers_fixed):
              for n, v in pp.named_parameters():
                 v.grad.data[self.is_fixed[i]] = 0 # torch.zeros_like(v.grad.data[self.is_fixed[i]])
                 if self.current_epoch == self.max_epochs-1:
                     self.update_gradient_data_tracker(i, v.grad)
                 i += 1


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.vaccuracy(F.softmax(y_hat, dim =1), y)
        self.log('val_loss', loss, logger = True, prog_bar=True, sync_dist=True, on_epoch=True, on_step=True)
        return loss 

    def validation_epoch_end(self, o):
        self.log('validation_acc_epoch', self.vaccuracy.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.ttaccuracy(F.softmax(y_hat, dim =1), y)
        top_5_acc = self.tt5accuracy(F.softmax(y_hat, dim =1), y)
        self.log('test_loss', loss, logger=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('test_acc', acc, logger=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('test_acc5', top_5_acc, logger=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss

