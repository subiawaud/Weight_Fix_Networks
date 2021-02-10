import os
import itertools

percentages = ["0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
distances_allowed = ["0.005"]
first_epochs = [0]
rest_epochs = [15]
bits = ['pow_2_add']
<<<<<<< HEAD:experiment_bash_run_cifar.py
gammas = [0.05, 0, 0.025, 0.1, 0.2]
=======
gammas = [0.05]
>>>>>>> 4941fcafc2636f703acc403e3bee5e6f0411b47d:experiment_bash_run.py
#gammas = [0]
#model = 'allconv'
zeros = [2**-7] # [2**-x for x in range(6, 9)]
model = 'resnet'
bn = True

params = list(itertools.product(*[percentages, distances_allowed, first_epochs, rest_epochs, bits, gammas, zeros]))
for param_set in params:
         percentage, da, f_e, r_e, b, gamma,z = param_set
         script_test =f'--distance_allowed {da} --regularistion_ratio {gamma} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --cluster_bit_fix {b} --name "exp_1" --model {model} --zero_distance {z} --bn_inc {bn}'
         os.system('sbatch script.sh '+ script_test)
