import os
import itertools

percentages = ["0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
distances_allowed = ["0.01 0.02 0.03 0.04 0.05"]
first_epochs = [0]
rest_epochs = [5]
bits = 'pow_2_add'
gammas = [0.8, 0.4, 0.2, 0.1, 0]#, 0, 0.025, 0.1, 0.2]
zeros = [2**-8] # 2**-8, 2**-9] # [2**-x for x in range(6, 9)]
model = 'mobilenet'
bn = [1.0, 0.0]
data = 'cifar10'
resume = 0.0
params = list(itertools.product(*[percentages, distances_allowed, first_epochs, rest_epochs, gammas, zeros, bn]))
for param_set in params:
         percentage, da, f_e, r_e, gamma,z, b = param_set
         script_test =f'--distance_allowed {da} --regularistion_ratio {gamma} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --cluster_bit_fix {bits} --name "exp_1" --model {model} --zero_distance {z} --bn_inc {b} --data {data} --resume {resume}'
         os.system('sbatch script_gtx.sh '+ script_test)
