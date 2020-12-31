import torch
from Models.All_Conv_4 import All_Conv_4
from Datasets import cifar10, mnist
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

def run_experiment(experiment_name, model, data, first_last_epochs, rest_epochs, percentages,distance_limit, distance_change, bits,t, gamma,  encourage_plus_one_cluster):
    experiment_name = f'e={experiment_name}-m={model.name}-d={data.name}-t={t}-g={gamma}-p1c={encourage_plus_one_cluster}-dl={distance_limit}-dc={distance_change}-cb={bits}-e1={first_last_epochs}-fe={rest_epochs}'
    dr = f'{os.getcwd()}/experiments/{experiment_name}'
    if not os.path.exists(dr):
        os.makedirs(dr)
    torch.save(model, dr + '/initial_model')
    outer_logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'joined',
                name = experiment_name
                )
    model.outer_logger = outer_logger


    for x in percentages:
        print('percentage of weights ', x)

        logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'iteration{x}',
                name = experiment_name
                )
        if x == percentages[0]: # or x == (iterations - 1):
          epochs = first_last_epochs

        else:
          epochs = rest_epochs

        model.reset_optim(epochs)
        trainer = pl.Trainer(gpus=1, max_epochs = epochs, logger = logger, num_sanity_val_steps = 0)
        trainer.fit(model, data)
        trainer.save_checkpoint(f'{os.getcwd()}/experiments/{experiment_name}/iteration_{x}_final_model')
        print(f'iteration{x},{experiment_name}')
        trainer.test()
        model.percentage_fixed = x
        model.cluster_prune(model.clusters)
        model.fixing_iteration +=1
        model.reset_weights()

    logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'final',
                name = experiment_name
                )
    trainer = pl.Trainer(gpus=1, max_epochs = 0, logger = logger, num_sanity_val_steps = 0)
    trainer.fit(model, data)
    trainer.save_checkpoint(f'{os.getcwd()}/experiments/{experiment_name}/complete_final_model')
    model.print_the_number_of_unique_params()
    print('Final test')
    trainer.test(ckpt_path=None)

def main(args):
    cifar = cifar10.CIFAR10DataModule()
    for c in args.change_to_try:
        for lim in args.limits_to_try:
            model = All_Conv_4(args.fixing_epochs)
            model.set_up(args.bits, lim, c, len(args.percentages), args.t, args.gamma, args.encourage_plus_one_cluster)
            model.flatten_is_fixed()
            run_experiment('set_1', model, cifar,args.first_epoch, args.fixing_epochs, args.percentages, lim, c, args.bits, args.t, args.gamma, args.encourage_plus_one_cluster)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limits_to_try',  nargs='+', type=float, default = [0.001])
    parser.add_argument('--change_to_try', nargs='+', type=float, default = [0.0002])
    parser.add_argument('--percentages', nargs='+', type=float, default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999,  1.0])
    parser.add_argument('--first_epoch', type=int, default = 25)
    parser.add_argument('--fixing_epochs', type=int, default = 5)
    parser.add_argument('--bits', default = 'pow_2')
    parser.add_argument('--name', default = "testing")
    parser.add_argument('--encourage_plus_one_cluster', default = True, type= bool)
    parser.add_argument('--gamma', default = 5, type=float)
    parser.add_argument('--t', default = 0.50, type = float)
    args = parser.parse_args()
    print(args)
    main(args)
