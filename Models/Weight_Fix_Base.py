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
from pytorch_lightning.metrics import functional as FM
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



class Weight_Fix_Base(pl.LightningModule):
    __metaclass_= abc.ABCMeta # to allow for abstract method implementations
    def __init__(self):
        super(Weight_Fix_Base, self).__init__()
        self.current_fixing_iteration = 0 # which iteration of the fixing algorithm are we at
        self.name = 'Base'  # object name
        self.tracking_gradients = False
        self.percentage_fixed = 0
        self.layers_fixed = None

    def reset_optim(self, max_epochs):
        self.max_epochs = max_epochs
        self.set_optim(max_epochs)

    def set_loggers(self, inner, outer):
        self.metric_logger.set_loggers(inner, outer)

    def set_up(self, distance_calculation_type, cluster_bit_fix, smallest_distance_allowed, number_of_fixing_iterations, regularisation_ratio, how_many_iterations_not_regularised, zero_distance, bn_inc):
        if bn_inc:
                print('BATCH NORM included')
                self.layers_fixed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm2d, nn.BatchNorm1d)
        else:
                self.layers_fixed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        self.parameter_iterator = Parameter_Iterator(self, self.layers_fixed)
        self.set_layer_shapes()
        self.set_up_fixed_weight_array()
        self.set_inital_weights()
        self.calculation_type = distance_calculation_type
        self.distance_calculator = Distance_Calculation(self.calculation_type)
        self.flattener = Flattener(self.parameter_iterator, self.is_fixed)
        self.cluster_determinator = Cluster_Determination(self.distance_calculator, self, self.is_fixed, self.calculation_type, self.layer_shapes, self.flattener, zero_distance)
        self.metric_logger = Metric_Capture(self)
        self.converter = Converter(cluster_bit_fix, distance_calculation_type, zero_distance)
        self.smallest_distance_allowed = smallest_distance_allowed
        self.cluster_bit_fix = cluster_bit_fix
        self.number_of_fixing_iterations = number_of_fixing_iterations
        self.how_many_iterations_not_regularised = how_many_iterations_not_regularised
        self.centroid_list = torch.Tensor([[0]]).to(self.device)
        self.regularisation_ratio = regularisation_ratio
        self.number_of_clusters = 1

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

    def update_results(self, exp_name, orig_acc, orig_entropy, orig_params, test_acc, fixing_epochs, data_name, zd):
        fixed_params = self.get_number_of_u_params()
        fixed_entropy, fixed_entropy_non_zero = self.get_weight_entropy()
        self.metric_logger.write_to_results_file(exp_name, self.name, self.regularisation_ratio,
        self.smallest_distance_allowed, fixing_epochs, orig_acc, orig_entropy, orig_params, test_acc, fixed_entropy, fixed_entropy_non_zero, fixed_params, data_name, zd)


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
            return [self.optim], [scheduler]
        else:
            print('no scheduler')
            return [self.optim]

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
        loss = ce + cluster_error
        result = pl.TrainResult(loss)
        acc = FM.accuracy(y_hat, y)
        result.log_dict({'train_acc':acc, 'train_loss':loss}, prog_bar=True, logger = False)
        return result

    def training_epoch_end(self, outputs):
        accuracy = torch.stack([outputs['train_acc']]).mean()
        loss = torch.stack([outputs['train_loss']]).mean()
        self.metric_logger.train_log(loss, accuracy, self.current_epoch)
        tensorboard_logs = {'Loss':loss, 'Accuracy': accuracy}
        epoch_dict = {'loss':loss, 'acc':accuracy}
        return epoch_dict

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

    def register_hooks(self):
        handles = []
        hooks = {}
        for name, layer in self.named_modules():
                hooks[name] = CaptureOutputs()
                handles.append(layer.register_forward_hook(hooks[name]))
        return handles, hooks

    def pass_through_training_data(self):
        train_loader = self.train_dataloader()
        for id, (x, y) in enumerate(train_loader):
            self(x.to(self.device))
            return

    def summarise_hooks(self, hooks):
        for k, v in hooks.items():
            v.summarise()

    def remove_handles(self, hooks):
        for h in hooks:
            h.remove()

    def forward_pass_with_hook(self):
        handles, hooks = self.register_hooks()
        self.pass_through_training_data()
        self.summarise_hooks(hooks)
        self.remove_handles(handles)

    def determine_which_weights_are_newly_fixed(self, idx, flattened_network):
        currently_fixed_indicies = np.argwhere(flattened_network.cpu())
        return np.setdiff1d(np.union1d(idx, currently_fixed_indicies), np.intersect1d(idx, currently_fixed_indicies))

    def calculate_threshold_value(self, distances_of_newly_fixed):
        # so I was doing this wrong??? shouldn't it be mean(abs(distances))
        return torch.mean(distances_of_newly_fixed) + 1*torch.std(distances_of_newly_fixed)


    def calculate_allowable_distance(self):
        a = max(self.smallest_distance_allowed, self.smallest_distance_allowed + self.smallest_distance_allowed*((self.number_of_fixing_iterations - self.current_fixing_iteration)/10))
        if self.number_of_clusters >= 100: # to stop out of memory issues
            print('ABORT')
            a *= 2
        return a

    def determine_which_weights_from_layers_should_be_clustered(self, closest_cluster_distances, allowable_distance, percentage):
        if percentage != 1.0:
            idx = self.cluster_determinator.select_layer_wise(closest_cluster_distances, allowable_distance, percentage)
        else:
            return range(number_fixed - 1)

    def calculate_how_many_to_fix(self, weights, percentage):
        return int(round(len(weights)*percentage, 2))

    def threshold_breached_handler(self, weights, percentage, quantised_weights):
         self.number_of_clusters += 1
         if (self.number_of_clusters == 5 or (self.number_of_clusters % 5 == 0 and self.number_of_clusters >= 8)) and self.cluster_bit_fix == "pow_2_add":
                 self.converter.increase_pow_2_level()
                 return self.apply_clustering_to_network()
         return self.apply_clustering_to_network(quantised_weights)

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
            self.fixed_weights[i] = clustered_weights[start:count].reshape(s).to(self.device)
            fixed += torch.sum(self.is_fixed[i])
        print('The number fixed is ', fixed.cpu().numpy())


    def print_unique_params(self):
        return self.parameter_iterator.iteratate_all_parameters_and_apply(self.metric_logger.print_the_number_of_unique_params)

    def apply_clustering_to_network(self, quantised_weights = None):
        weights = self.flattener.flatten_network_tensor()
        percentage = self.percentage_fixed
        number_fixed = self.calculate_how_many_to_fix(weights, percentage)
        if quantised_weights is None:
            quantised_weights = self.converter.round_to_precision(weights, self.calculate_allowable_distance())
        centroids, centroid_to_regularise_to = self.cluster_determinator.find_closest_centroids(quantised_weights, self.number_of_clusters)
        self.centroid_to_regularise_to = centroid_to_regularise_to
        closest_cluster_distance, closest_cluster_list = self.cluster_determinator.closest_cluster(weights, centroids, self.current_fixing_iteration)
        self.determine_which_weights_from_layers_should_be_clustered(closest_cluster_distance, percentage, number_fixed)
        if percentage != 1.0:
            idx = self.cluster_determinator.select_layer_wise(closest_cluster_distance, self.smallest_distance_allowed, percentage)
        else:
           idx = range(number_fixed)
        newly_fixed = self.determine_which_weights_are_newly_fixed(idx, self.flattener.flatten_standard(self.is_fixed))
        threshold_val = self.calculate_threshold_value(closest_cluster_distance[newly_fixed])
        print('current Val ', threshold_val)
        if threshold_val > self.calculate_allowable_distance() and self.current_fixing_iteration > 1:
             print('centroids are', centroids)
             print('number of centroids', self.number_of_clusters)
             return self.threshold_breached_handler(weights, percentage, quantised_weights)
        weights, clustered = self.gather_assigned_clusters(centroids, idx, closest_cluster_list, weights)
        self.metric_logger.summarise_clusters_selected(centroids, closest_cluster_distance[newly_fixed],threshold_val, self.smallest_distance_allowed, self.current_fixing_iteration)
        self.assign_weights_to_clusters(weights, clustered)

    def update_gradient_data_tracker(self, i, grad):
        if self.tracking_gradients:
            self.grads[i] = np.abs(grad.data.cpu().detach().numpy())

    def on_after_backward(self):
        i = 0
        for n, pp in self.named_modules():
          if isinstance(pp, self.layers_fixed):
              for n, v in pp.named_parameters():
                 v.grad.data[torch.where(self.is_fixed[i])] = torch.zeros_like(v.grad.data[torch.where(self.is_fixed[i])])
                 if self.current_epoch == self.max_epochs-1:
                     self.update_gradient_data_tracker(i, v.grad)
                 i += 1


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
        self.metric_logger.validation_log(loss, accuracy, self.current_epoch)
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
        self.metric_logger.test_log(loss, accuracy, self.percentage_fixed, self.current_fixing_iteration)
        result = pl.EvalResult()
        result.log_dict({'test_acc':accuracy, 'test_loss':loss}, prog_bar=True, logger=False)
        return result
