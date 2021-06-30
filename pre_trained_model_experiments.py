import torch
import time
from Models.Pretrained_Model_Template import Pretrained_Model_Template
from Datasets import cifar10, mnist, imagenet, coco
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torchvision.models as models
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PyTorch_CIFAR10.cifar10_models import *
from Models.All_Conv_4 import All_Conv_4
import re
import numpy as np


def run_experiment(experiment_name, model, data, first_last_epochs, rest_epochs, epoch_increment, percentages,distance_allowed, cluster_bit_fix,regularistion_ratio, model_name, non_regd, zd, bn, check_point = 0.0):
    experiment_name = f'e={experiment_name}-m={model_name}-d={data.name}-rr={regularistion_ratio}-d_a={distance_allowed}-cb={cluster_bit_fix}-e1={first_last_epochs}-fe={rest_epochs}-ei={epoch_increment}-nr={non_regd}-zd={zd}-bn={bn}'
    checkpoint_address = '/scratch/cc2u18/Weight_Fix_Networks/'
    dr = f'{checkpoint_address}/experiments/{experiment_name}'
    if not os.path.exists(dr):
        try:
            os.makedirs(dr)
        except:
            print('drive already exists')

    torch.save(model, dr + '/initial_model')
    outer_logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'joined',
                name = experiment_name
                )
#    else:
#        it = []
#        for f in os.listdir(dr):
#            if f.startswith("iteration"):
#               it.append(float(re.findall('\d+\.\d+',f)[0]))
#         current = np.max(np.array(it))
#         print('Current is ', current)
#         percentages = percentages[percentages >= current]
#         model = load_from_checkpoint('fix me')
    if check_point > 0.0:
        inner_logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'baseline',
                name = experiment_name
                )
        epochs = first_last_epochs
        trainer = pl.Trainer(gpus=-1, precision=16, gradient_clip_val = 0.5, accelerator='ddp', max_epochs = epochs, logger = inner_logger, num_sanity_val_steps = 0, checkpoint_callback=False)
        model.set_loggers(inner_logger, outer_logger)
        model.reset_optim(0)
        trainer.fit(model, data)
        print('GETTING THE ORIGINAL MODEL DATA')
        orig_acc = trainer.test(model)[0]['test_acc_epoch']
        orig_entropy = model.get_weight_entropy()
        orig_params = model.get_number_of_u_params()
        model = model.load_from_checkpoint(checkpoint_path=f'{checkpoint_address}/experiments/{experiment_name}/iteration_{check_point}_final_model',max_epochs=model.max_epochs, original_model=model.pretrained, data_module=model.data_module, lr= model.lr, use_sched=model.use_sched,opt= model.opt)
        model.set_up('relative', cluster_bit_fix, distance_allowed, len(percentages), regularistion_ratio, non_regd, zd, bn)
        model.flatten_is_fixed()
    for i,x in enumerate(percentages):
        if x < check_point:
                model.current_fixing_iteration +=1
                print('skipping ', x)
                continue
        print('percentage of weights ', x)
        inner_logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'iteration{x}',
                name = experiment_name
                )
        if x == percentages[0] or x == check_point: # or x == percentages[-1]: # or x == (iterations - 1):
          epochs = first_last_epochs
        else:
          epochs = rest_epochs + (i//2)*epoch_increment
        model.set_loggers(inner_logger, outer_logger)
        model.reset_optim(epochs)

        trainer = pl.Trainer(gpus=1, max_epochs = epochs, gradient_clip_val=1, logger = inner_logger, num_sanity_val_steps = 0,
                             checkpoint_callback=False)
        trainer.fit(model, data)


        if x == percentages[0]:
            orig_acc = trainer.test(model)[0]['test_acc']
            orig_entropy = model.get_weight_entropy()
            orig_params = model.get_number_of_u_params()
        else:
            acc = trainer.test(model)
            print(acc)
        model.percentage_fixed = x
        model.apply_clustering_to_network()
        model.current_fixing_iteration +=1
        model.reset_weights()
        try:
                trainer.save_checkpoint(f'{checkpoint_address}/experiments/{experiment_name}/iteration_{x}_final_model')
                model.save_clusters()
        except:
                print('SAVED failed')
    logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'final',
                name = experiment_name
                )
    time.sleep(30)
    model = model.load_from_checkpoint(checkpoint_path=f'{checkpoint_address}/experiments/{experiment_name}/iteration_1.0_final_model',max_epochs=model.max_epochs, original_model=model.pretrained, data_module=model.data_module, lr= model.lr, use_sched=model.use_sched,opt= model.opt)
    model.set_up('relative', cluster_bit_fix, distance_allowed, len(percentages), regularistion_ratio, non_regd, zd, bn)
    model.reset_optim(epochs)
    trainer = pl.Trainer(gpus=1, gradient_clip_val = 0.5, accelerator='ddp', max_epochs = 0, logger = inner_logger, num_sanity_val_steps = 0, checkpoint_callback=False)
    trainer.fit(model, data)
    model.print_unique_params()
    print('The final test is running')
    model.eval()
    acc = trainer.test(model, ckpt_path=None)[0]['test_acc_epoch']
    trainer.save_checkpoint(f'{checkpoint_address}/experiments/{experiment_name}/complete_final_model')
    torch.save(model.state_dict(), f'{checkpoint_address}/experiments/{experiment_name}/complete_final_model_state_dict')
    model.update_results(experiment_name, orig_acc, orig_entropy, orig_params, acc, rest_epochs, data.name, zd)


def get_model(model, data):
    print(model, data)
    use_sched = True
    if model == 'conv4':
        model = All_Conv_4()
        use_sched = False
        lr = 3e-4
        model = model.load_from_checkpoint(checkpoint_path="PyTorch_CIFAR10/cifar10_models/state_dicts/all_conv4")
        model.name = 'conv4'
        return lr, use_sched, model, 'ADAM'
    if model == 'resnet' and data == 'cifar10':

        lr = 0.00002

        m = resnet18(pretrained=True)
        m.name = model
        return lr, use_sched, m, 'ADAM'
    if model == 'resnet50' and data == 'coco':
        lr = 0.00002
        m = models.segmentation.fcn_resnet50(pretrained=True)
        m.name = 'resnet50_coco'
        return lr, use_sched, m, 'ADAM'

    if model == 'resnet' and data == 'imnet':
        lr = 0.00002
        model = models.resnet18(pretrained=False)
        model.name = 'resnet18'
        model.load_state_dict(torch.load("PyTorch_ImNet/resnet18"))
        return lr, use_sched, model, 'ADAM'

    if model == 'resnet50': # and data == 'imnet':
        lr = 0.00002
        model = models.resnet50(pretrained=False)
        model.name = 'resnet50'
        model.load_state_dict(torch.load("PyTorch_ImNet/resnet50"))
        return lr, use_sched, model, 'ADAM'

    if model == 'mobilenet' and data = 'cifar10':

        lr = 0.00002
        m = mobilenet_v2(pretrained=True)
        m.name = model
        return lr, use_sched, m, 'ADAM'

def determine_dataset(data_arg):
    if data_arg == 'cifar10':
        return cifar10.CIFAR10DataModule()
    elif data_arg == 'imnet':
        return imagenet.ImageNet_Module()
    elif data_arg == 'coco':
        return coco.CocoDataModule()

def main(args):
    data = determine_dataset(args.dataset)
    data.setup()
    for d_a in args.distance_allowed:
            lr, use_sched, model, opt = get_model(args.model, args.dataset)
            model = Pretrained_Model_Template(model, args.fixing_epochs + 1, data, lr, use_sched, opt)
            model.set_up(args.distance_calculation_type, args.cluster_bit_fix, d_a, len(args.percentages), args.regularistion_ratio, args.non_regd, args.zero_distance, args.bn_inc)
            model.flatten_is_fixed()
            run_experiment('set_1', model, data,args.first_epoch, args.fixing_epochs, args.epoch_increment, args.percentages, d_a, args.cluster_bit_fix, args.regularistion_ratio, args.model, args.non_regd, args.zero_distance, args.bn_inc > 0.1, args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_allowed',  nargs='+', type=float, default = [0.075]) #0.1, 0.15, 0.2, 0.25, 0.3
    parser.add_argument('--percentages', nargs='+', type=float, default = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999,  1.0])
    parser.add_argument('--first_epoch', type=int, default =0)
    parser.add_argument('--fixing_epochs', type=int, default = 5)
    parser.add_argument('--epoch_increment', type=int, default = 0)
    parser.add_argument('--cluster_bit_fix', default = 'pow_2_add')
    parser.add_argument('--name', default = "testing_relative")
    parser.add_argument('--distance_calculation_type', default = "relative")
    parser.add_argument('--regularistion_ratio', default = 0.1, type=float) #0.075, 0.05, 0.025, 0.01, 0.1
    parser.add_argument('--model', default = 'resnet50')
    parser.add_argument('--non_regd', default = 0, type=float)
    parser.add_argument('--dataset', default = 'cifar10')
    parser.add_argument('--zero_distance', default = 2**-12, type=float)
    parser.add_argument('--bn_inc', default=0.0, type=float)
    parser.add_argument('--resume',default=0.0, type=float)
    args = parser.parse_args()
    print('This is the args ', args)
    main(args)
