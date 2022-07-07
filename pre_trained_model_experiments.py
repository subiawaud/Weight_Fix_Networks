import torch
import time
from pytorch_lightning.plugins import DDPPlugin
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
from Models.All_Conv_4 import All_Conv_4
import re
import numpy as np

LOCAL = False

def run_the_model_with_no_training(outer_logger, model, data, address, exp_n):
        inner_logger = TensorBoardLogger(
                save_dir = f'{address}/experiments/',
                version = f'baseline',
                name = exp_n
                )
        trainer = pl.Trainer(gpus=-1, precision=16, gradient_clip_val = 0.5, accelerator='ddp', max_epochs = 0, logger = inner_logger, num_sanity_val_steps = 0, checkpoint_callback=False,plugins=DDPPlugin(find_unused_parameters=False))
        model.reset_optim(0, inner_logger, outer_logger)
        trainer.fit(model, data)
        return trainer, model

def make_address(dr):
    if not os.path.exists(dr):
        try:
            os.makedirs(dr)
        except:
            print('drive already exists')

     
def run_experiment(experiment_name, model, data, first_last_epochs, rest_epochs, percentages,distance_allowed, regularistion_ratio, model_name, zd, bn, check_point = 0.0, cal_type='relative'):
    experiment_name = f'e={experiment_name}-m={model_name}-d={data.name}-rr={regularistion_ratio}-d_a={distance_allowed}-e1={first_last_epochs}-fe={rest_epochs}-zd={zd}-bn={bn}-ct={cal_type}'
    if not LOCAL:
       checkpoint_address = '/scratch/cc2u18/Weight_Fix_Networks/'
    else:
       checkpoint_address = '/'
    dr = f'{checkpoint_address}/experiments/{experiment_name}'
    make_address(dr)

    torch.save(model, dr + '/initial_model')
    outer_logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'joined',
                name = experiment_name
                )

    if check_point > 0.0:
        trainer, model  = run_the_model_with_no_training(outer_logger, model, data, checkpoint_address, experiment_name)
        orig_acc = trainer.test(model)[0]['test_acc_epoch']
        orig_entropy = model.get_weight_entropy()
        orig_params = model.get_number_of_u_params()
        model = model.load_from_checkpoint(checkpoint_path=f'{checkpoint_address}/experiments/{experiment_name}/iteration_{check_point}_final_model',max_epochs=model.max_epochs, original_model=model.pretrained, data_module=model.data_module, lr= model.lr, scheduler=model.scheduler,opt= model.opt)
        model.set_up(distance_allowed, len(percentages), regularistion_ratio, zd, bn, cal_type)

    for i,x in enumerate(percentages):
        if x < check_point: # skip this iteration 
                model.current_fixing_iteration +=1
                continue

        print('percentage of weights ', x)
        inner_logger = TensorBoardLogger(
                save_dir = f'{checkpoint_address}/experiments/',
                version = f'iteration{x}',
                name = experiment_name
                )
        if x == percentages[0] or x == check_point: # first and last iterations have different number of epochs to the rest
          epochs = first_last_epochs
        else:
          epochs = rest_epochs 
        model.reset_optim(epochs, inner_logger, outer_logger)


        if torch.cuda.is_available():
               gpus = -1
               accelerator='ddp' 
        else:
               gpus=0
               accelerator=None
        trainer = pl.Trainer(gpus=gpus, max_epochs = epochs, gradient_clip_val=0.5, logger = inner_logger, num_sanity_val_steps = 0,
                             checkpoint_callback=False, plugins=DDPPlugin(find_unused_parameters=False), accelerator=accelerator)

        trainer.fit(model, data)

        if x == percentages[0]:
            orig_acc = trainer.test(model)[0]['test_acc']
            orig_entropy = model.get_weight_entropy()
            orig_params = model.get_number_of_u_params()
        else:
            acc = trainer.test(model)
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
    time.sleep(60)
    model = model.load_from_checkpoint(checkpoint_path=f'{checkpoint_address}/experiments/{experiment_name}/iteration_1.0_final_model',max_epochs=model.max_epochs, original_model=model.pretrained, data_module=model.data_module, lr= model.lr, scheduler=model.scheduler,opt= model.opt)

    model.set_up(distance_allowed, len(percentages), regularistion_ratio, zd, bn, cal_type)
    model.reset_optim(epochs, logger, outer_logger)
    trainer = pl.Trainer(gpus=-1, gradient_clip_val = 0.5, accelerator=accelerator, max_epochs = 0, logger = inner_logger, num_sanity_val_steps = 0, checkpoint_callback=False)
    trainer.fit(model, data)
    model.print_unique_params()
    model.eval()
    acc = trainer.test(model, ckpt_path=None)[0]['test_acc_epoch']
    trainer.save_checkpoint(f'{checkpoint_address}/experiments/{experiment_name}/complete_final_model')
    torch.save(model.state_dict(), f'{checkpoint_address}/experiments/{experiment_name}/complete_final_model_state_dict')
    model.update_results(experiment_name, orig_acc, orig_entropy, orig_params, acc, rest_epochs, data.name, zd, bn)


def get_model(model_name, data):
    """ Here is where the models are defined, if you would like to use a new model, you can insert it into here """

    if model_name == 'conv4':
        model = All_Conv_4()
        model = model.load_from_checkpoint(checkpoint_path="Pretrained_Models/PyTorch_CIFAR10/cifar10_models/state_dicts/all_conv4")

    if model_name == 'resnet18' and data == 'cifar10':
        model = resnet18(pretrained=True)

    if model_name == 'resnet34':# and data == 'imnet':
        model = models.resnet34(pretrained=True)
#        torch.save(model.state_dict(), "Pretrained_Models/PyTorch_ImNet/resnet34")
#        model.load_state_dict(torch.load("Pretrained_Models/PyTorch_ImNet/resnet34"))

    if model_name == 'googlenet':
        model = models.googlenet(pretrained=True, aux_logits = False)
#        torch.save(model.state_dict(), 'Pretrained_Models/PyTorch_ImNet/googlenet')
#        model.load_state_dict(torch.load("Pretrained_Models/PyTorch_ImNet/googlenet"))
       
    if model_name == 'resnet18' and data == 'imnet':
        model = models.resnet18(pretrained=True)
        model.load_state_dict(torch.load("Pretrained_Models/PyTorch_ImNet/resnet18"))

    if model_name == 'resnet50': # and data == 'imnet':
        model = models.resnet50(pretrained=True)
#        model.load_state_dict(torch.load("Pretrained_Models/PyTorch_ImNet/resnet50"))

    if model_name == 'mobilenet' and data == 'imnet':
        model = models.mobilenet_v2(pretrained=True)
        #torch.save(model.state_dict(), 'Pretrained_Models/PyTorch_ImNet/mobilenet')
#        model.load_state_dict(torch.load("Pretrained_Models/PyTorch_ImNet/mobilenet"))

    if model_name == 'mobilenet' and data == 'cifar10':
        model = mobilenet_v2(pretrained=True)
    model.name = model_name
    return model

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
            model = get_model(args.model, args.dataset)
            model = Pretrained_Model_Template(model, args.fixing_epochs + 1, data, args.lr, args.scheduler, args.optimiser, args)
            model.set_up(d_a, len(args.percentages), args.regularistion_ratio, args.zero_distance, args.bn_inc, args.calculation_type)
            run_experiment(args.experiment_name, model, data,args.first_epoch, args.fixing_epochs, args.percentages, d_a,  args.regularistion_ratio, args.model, args.zero_distance, args.bn_inc > 0.1, args.resume, args.calculation_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_allowed',  nargs='+', type=float, default = [0.075]) 
    parser.add_argument('--percentages', nargs='+', type=float, default = [0.3, 0.6, 0.8, 0.9, 0.95, 0.975, 0.999,  1.0])
    parser.add_argument('--optimiser', default='ADAM')
    parser.add_argument('--experiment_name', default='retry')
    parser.add_argument('--scheduler', default='None')
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--first_epoch', type=int, default =0)
    parser.add_argument('--fixing_epochs', type=int, default = 3)
    parser.add_argument('--regularistion_ratio', default = 0.2, type=float) 
    parser.add_argument('--model', default = 'googlenet')
    parser.add_argument('--dataset', default = 'cifar10')
    parser.add_argument('--zero_distance', default = 2**-7, type=float)
    parser.add_argument('--bn_inc', default=0.0, type=float)
    parser.add_argument('--resume',default=0.0, type=float)
    parser.add_argument('--calculation_type',default='relative')
    args = parser.parse_args()
    main(args)
