import os
import itertools

percentages = ["0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
limits = ["0.001 0.0005 0.00025 0.002"]
changes = ["0.0001 0.000025 0.0002 0.00005"]
first_epochs = [40]
rest_epochs = [20, 5, 10]
bits = [32]
script_test =f'python all_conv_experiments.py --limits_to_try 0.002 0.0025 --change_to_try 0.0001 0.005 --percentages {percentages} --first_epoch 1 --fixing_epochs 1 --bits 16 --name "testing_argparse"'
ts = [1, 0.75, 0.5]
gammas = [0.5, 0.25, 0.1, 0]
encourage_plus_one_clusters = [True] #, False]

params = list(itertools.product(*[percentages, limits, changes, first_epochs, rest_epochs, bits, ts, gammas, encourage_plus_one_clusters]))
print(len(params))
for param_set in params:
         percentage, limit, c, f_e, r_e, b, t, gamma, epoc = param_set
         script_test =f'--encourage_plus_one_cluster {epoc} --t {t} --gamma {gamma} --limits_to_try {limit} --change_to_try {c} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --bits {b} --name "testing_argparse" '
         os.system('sbatch script.sh ' script_test)
