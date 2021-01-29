import os
import itertools

percentages = ["0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
distances_allowed = ["0.15 0.1 0.2 0.25 0.3 0.35"]
first_epochs = [0]
rest_epochs = [10, 20]
bits = ['pow_2_add']
gammas = [0.05, 0, 0.025, 0.075, 0.1, 0.15,  0.2, 0.25]
#gammas = [0]
model = 'allconv'

params = list(itertools.product(*[percentages, distances_allowed, first_epochs, rest_epochs, bits, gammas]))
for param_set in params:
         percentage, da, f_e, r_e, b, gamma, = param_set
         script_test =f'--distance_allowed {da} --regularistion_ratio {gamma} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --cluster_bit_fix {b} --name "exp_1" '
         os.system('sbatch -A ecsstaff script.sh '+ script_test)
