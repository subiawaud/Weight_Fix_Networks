import torch
from functools import partial
import numpy as np
import pandas as pd
import os.path

class Metric_Capture():
    def __init__(self, model, inner_logger = None, outer_logger = None):
        self.model = model
        self.inner_logger = inner_logger
        self.outer_logger = outer_logger
        self.results_file = 'Results.csv'
        self.check_results_file_exists()

    def get_results_columns(self):
        return np.array(['data_name', 'experiment_name', 'model_name', 'BN', 'regularisation_ratio',
         'distance_allowed',  'fixing_epochs',  'orig_acc', 'orig_entropy', 'orig_params',
                'compressed_acc', 'compressed_entropy','compressed_entropy_non_zero', 'unique_params', 'zero_distance'])

    def check_results_file_exists(self):
        if not os.path.isfile(self.results_file):
            res_file = pd.DataFrame(columns = self.get_results_columns())
            res_file.to_csv(self.results_file, index=False)

    def set_inner_logger(self,inner):
        self.inner_logger = inner

    def set_outer_logger(self, outer):
        print('SETTTINg the outer', outer)
        self.outer_logger = outer
        print('the outer logger is ', self.outer_logger)

    def write_to_results_file(self, exp_name, model_name, regularistion_ratio,
    distance_allowed, fixing_epochs, orig_acc, orig_entropy, orig_params, compressed_acc, compressed_entropy, non_zero_entropy, unique_params, data_name, zero_distance, bn):
        d ={'data_name': data_name, 'experiment_name' : exp_name, 'model_name' : model_name, 'BN':bn, 'regularistion_ratio' : regularistion_ratio,
        'distance_allowed' : distance_allowed,  'fixing_epochs' : fixing_epochs,'orig_acc' : orig_acc, 'orig_entropy': orig_entropy, 'orig_params' :orig_params,
         'compressed_acc' : compressed_acc,  'compressed_entropy' : compressed_entropy, 'compressed_entropy_non_zero':non_zero_entropy,
        'unique_params' : unique_params, 'zero_distance' : zero_distance}
        new = pd.DataFrame.from_records([d])
        new.to_csv(self.results_file, mode='a', header=False, index=False)


    def summarise_max_distance(self, max_distance, iteration):
           self.outer_logger.experiment.add_scalar("Clusters/max_distance", max_distance, iteration)

    def summarise_mean_plus_std(self, val, iteration):
           self.outer_logger.experiment.add_scalar("Clusters/mean_dist+std", val, iteration)

    def summarise_distance_allowed(self, distance_allowed, iteration):
           self.outer_logger.experiment.add_scalar("Clusters/distance_allowed", distance_allowed, iteration)

    def summarise_distance_values(self, distances, iteration):
           self.outer_logger.experiment.add_scalar("Clusters/median_distance", torch.median(distances), iteration)
           self.outer_logger.experiment.add_scalar("Clusters/std_distance", torch.std(distances), iteration)
           self.outer_logger.experiment.add_scalar("Clusters/mean_distance", torch.mean(distances),iteration)

    def summarise_number_of_clusters(self, centroids, iteration):
        number = len(centroids[0])
        self.outer_logger.experiment.add_scalar("Clusters/number_of_clusters", number, iteration)

    def summarise_order_dist(self, order_dist, iteration):
        for i, o in enumerate(order_dist):
            self.outer_logger.experiment.add_scalar(f"Clusters/Order_{i}", int(o), iteration)
        
    def train_log(self, loss, acc, iteration):
           self.inner_logger.experiment.add_scalar("Loss/Train", loss, iteration)
           self.inner_logger.experiment.add_scalar("Accuracy/Train", acc, iteration)


    def validation_log(self, loss, acc, iteration):
           self.inner_logger.experiment.add_scalar("Loss/Val", loss, iteration)
           self.inner_logger.experiment.add_scalar("Accuracy/Val", acc, iteration)

    def test_log(self, loss, acc, percentage_fixed, iteration):
           print(self.outer_logger, ' This is the logger of loggers')
           self.outer_logger.experiment.add_scalar("Loss/Test", loss, iteration)
           self.outer_logger.experiment.add_scalar("Accuracy/Test", acc, iteration)
           self.outer_logger.experiment.add_scalar("Fixed_Percentage", percentage_fixed, iteration)


    def summarise_entropy(self, entropy, iteration):
         self.outer_logger.experiment.add_scalar('Clusters/Entropy', entropy, iteration)

    def summarise_clusters_selected(self, centroids, distances,mean_plus_std, distance_allowed, iteration, order_dist, entropy):
          self.summarise_max_distance(torch.max(distances), iteration)
          self.summarise_mean_plus_std(mean_plus_std, iteration)
          self.summarise_distance_allowed(distance_allowed, iteration)
          self.summarise_distance_values(distances, iteration)
          self.summarise_number_of_clusters(centroids, iteration)
          self.summarise_order_dist(order_dist, iteration)
          self.summarise_entropy(entropy, iteration)


    def split_bias_and_weight_histogram(self,label, name, p):
        w = p.weight
        b = p.bias
        p = p[torch.abs(params) > 0.00001]
        self.inner_logger.experiment.add_histogram(str(name) + '_weights', p, label)

    def custom_histogram_adder(self, label = None):
        if label is None:
            label = self.current_epoch
        function = partial(split_bias_and_weight_histogram, label)
        return function


    def layer_sparsity(self, params):
         with torch.no_grad():
              p = params[torch.abs(params) > 0.00001]
              number_of_zeros = params.numel() - p.numel()
              tot = params.numel()
              sparsity = number_of_zeros / tot
              return number_of_zeros, tot , sparsity

    def print_the_number_of_unique_params(self, name, p,  count = False):
                if count:
                    print(np.unique(p.data.cpu().detach().numpy(), return_counts = count))
                else:
                    print(len(np.unique(p.data.cpu().detach().numpy())))
