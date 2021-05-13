import os
import itertools


percentages = ["0.2 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
distances_allowed = ["0.05 0.1 0.15"] #0.075 0.05 0.025"]# 0.025 0.05 0.1"]
first_epochs = [0]
rest_epochs = [10]
bits = ['pow_2_add']
gammas = [0.075, 0.05, 0.025, 0.0]
#gammas = [0]
models = ['resnet']
zeros = [2**-12] # [2**-x for x in range(6, 9)]
dataset = 'cifar10'
bn = 0.0
epoch_increment = 0
iteration = 0.0
params = list(itertools.product(*[percentages, distances_allowed, first_epochs, rest_epochs, bits, gammas, zeros, models]))
print('NUMBER TO RUN', len(params))
for param_set in params:
         print(param_set)
         percentage, da, f_e, r_e, b, gamma,z, model = param_set
         script_test =f'--distance_allowed {da} --regularistion_ratio {gamma} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --epoch_increment {epoch_increment} --cluster_bit_fix {b} --name "bn_in" --model {model} --zero_distance {z} --bn_inc {bn} --data {dataset} --resume {iteration}'
         os.system('python pre_trained_model_experiments.py '+ script_test)
