import os
import itertools

#percentages = ["0.2 0.4 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.9975 0.999 1.0"]
percentages = ["0.3 0.6 0.8 0.9 0.95 0.975 0.99 0.9975 0.999 1.0"]

distances_allowed = ["0.0125"]# 0.025 0.05 0.1"]
first_epochs = [0]
rest_epochs = [0]
bits = ['pow_2_add']
gammas = [0.4]
#gammas = [0]
#model = 'allconv'
zeros = [2**-7] # [2**-x for x in range(6, 9)]
dataset = 'imnet'
models = ['resnet18', 'resnet34', 'resnet50', 'googlenet']#, 'mobilenet']#, 'resnet50']
bn = 1.0
epoch_increment = 0
iteration = 0.0
params = list(itertools.product(*[percentages, distances_allowed, first_epochs, rest_epochs, bits, gammas, zeros, models]))
for param_set in params:
         print(param_set)
         percentage, da, f_e, r_e, b, gamma,z, model = param_set
         script_test =f'--distance_allowed {da} --regularistion_ratio {gamma} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e}  --model {model} --zero_distance {z} --bn_inc {bn} --data {dataset} --resume {iteration}'
         os.system('sbatch script.sh '+ script_test)
