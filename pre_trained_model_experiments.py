import torch
from Models.Pretrained_Model_Template import Pretrained_Model_Template
from Datasets import cifar10, mnist
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torchvision.models as models
from PyTorch_CIFAR10.cifar10_models import *
from Models.All_Conv_4 import All_Conv_4


def run_experiment(experiment_name, model, data, first_last_epochs, rest_epochs, percentages,distance_allowed, cluster_bit_fix,regularistion_ratio, model_name, non_regd, zd):
    experiment_name = f'e={experiment_name}-m={model_name}-d={data.name}-rr={regularistion_ratio}-d_a={distance_allowed}-cb={cluster_bit_fix}-e1={first_last_epochs}-fe={rest_epochs}-nr={non_regd}-zd={zd}'
    dr = f'{os.getcwd()}/experiments/{experiment_name}'
    if not os.path.exists(dr):
        os.makedirs(dr)
    torch.save(model, dr + '/initial_model')
    outer_logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'joined',
                name = experiment_name
                )
    for x in percentages:
        print('percentage of weights ', x)
        inner_logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'iteration{x}',
                name = experiment_name
                )
        if x == percentages[0]: # or x == percentages[-1]: # or x == (iterations - 1):
          epochs = first_last_epochs
        else:
          epochs = rest_epochs
        model.set_loggers(inner_logger, outer_logger)
        model.reset_optim(epochs)
        trainer = pl.Trainer(gpus=1, max_epochs = epochs, logger = inner_logger, num_sanity_val_steps = 0, checkpoint_callback=False)
        trainer.fit(model, data)
        if x == percentages[0]:
            orig_acc = trainer.test(model)[0]['test_acc']
            orig_entropy = model.get_weight_entropy()
            orig_params = model.get_number_of_u_params()

#        trainer.save_checkpoint(f'{os.getcwd()}/experiments/{experiment_name}/iteration_{x}_final_model')
        trainer.test(model)
        model.percentage_fixed = x
        model.apply_clustering_to_network()
        model.current_fixing_iteration +=1
        model.reset_weights()

    logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/',
                version = f'final',
                name = experiment_name
                )
    model.set_loggers(logger, outer_logger)
    trainer = pl.Trainer(gpus=1, max_epochs = 0, logger = logger, num_sanity_val_steps = 0)
    model.reset_optim(epochs)
    trainer.fit(model, data)
    trainer.save_checkpoint(f'{os.getcwd()}/experiments/{experiment_name}/complete_final_model')
    model.print_unique_params()
    acc = trainer.test(model, ckpt_path=None)[0]['test_acc']
    model.update_results(experiment_name, orig_acc, orig_entropy, orig_params, acc, rest_epochs, data.name, zd)

def get_model(model):
    use_sched = True
    if model == 'conv4':
        model = All_Conv_4()
        use_sched = False
        lr = 3e-4
        model = model.load_from_checkpoint(checkpoint_path="PyTorch_CIFAR10/cifar10_models/state_dicts/all_conv4")
        return lr, use_sched, model, 'ADAM'
    if model == 'resnet':
        lr = 0.00002
        return lr, use_sched, resnet18(pretrained = True), 'ADAM'
    if model == 'mobilenet':
        lr = 0.00002
        return lr, use_sched, mobilenet_v2(pretrained = True), 'ADAM'
    if model == 'vgg':
        lr = 0.00002
        return lr, use_sched, vgg11_bn(pretrained = True), 'ADAM'

def main(args):
    cifar = cifar10.CIFAR10DataModule()
    cifar.setup()
    for d_a in args.distance_allowed:
            lr, use_sched, model, opt = get_model(args.model)
            model = Pretrained_Model_Template(model, args.fixing_epochs + 1, cifar, lr, use_sched, opt)
            model.set_up(args.distance_calculation_type, args.cluster_bit_fix, d_a, len(args.percentages), args.regularistion_ratio, args.non_regd, args.zero_distance)
            model.flatten_is_fixed()
            run_experiment('set_1', model, cifar,args.first_epoch, args.fixing_epochs, args.percentages, d_a, args.cluster_bit_fix, args.regularistion_ratio, args.model, args.non_regd, args.zero_distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_allowed',  nargs='+', type=float, default = [0.05]) #0.1, 0.15, 0.2, 0.25, 0.3
    parser.add_argument('--percentages', nargs='+', type=float, default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999,  1.0])
    parser.add_argument('--first_epoch', type=int, default = 0)
    parser.add_argument('--fixing_epochs', type=int, default = 10)
    parser.add_argument('--cluster_bit_fix', default = 'pow_2_add')
    parser.add_argument('--name', default = "testing_relative")
    parser.add_argument('--distance_calculation_type', default = "relative")
    parser.add_argument('--regularistion_ratio', default = 0.05, type=float) #0.075, 0.05, 0.025, 0.01, 0.1
    parser.add_argument('--model', default = 'conv4')
    parser.add_argument('--non_regd', default = 0, type=float)
    parser.add_argument('--zero_distance', default = 2**-6, type=float)
    args = parser.parse_args()
    print(args)
    main(args)
