import os


percentages = ["0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.975 0.99 0.995 0.999 1.0"]
limits = ["0.002 0.003 0.025 0.0015 0.001"]
changes = ["0.0001 0.00025 0.0005 0.00005"]
first_epochs = [5, 10, 20, 40, 80]
rest_epochs = [2, 5, 10, 15, 20]
bits = [32]
script_test =f'python all_conv_experiments.py --limits_to_try 0.002 0.0025 --change_to_try 0.0001 0.005 --percentages {percentages} --first_epoch 1 --fixing_epochs 1 --bits 16 --name "testing_argparse"'

for percentage in percentages:
        for limit in limits:
                for c in changes:
                        for f_e in first_epochs:
                                for r_e in rest_epochs:
                                        for b in bits:
                                            script_test =f'python all_conv_experiments.py --limits_to_try {limit} --change_to_try {c} --percentages {percentage} --first_epoch {f_e} --fixing_epochs {r_e} --bits {b} --name "testing_argparse" --weight_entropy_reg 0.1'
                                            os.system(script_test)
