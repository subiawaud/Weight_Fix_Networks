import torch
from Models import ResNet
from Datasets import cifar10, mnist
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger 
import os 

cifar = cifar10.CIFAR10DataModule()
mnist = mnist.MNISTDataModule()
number_cluster_bits = 16 

model = ResNet.resnet18()
model.set_up(number_cluster_bits)
model.reset_weights()
clusters = [2, 2, 2,2,6,14,16] 


def run_experiment(experiment_name, model, data, clusters):
    experiment_name = f'experiment={experiment_name}-model={model.name}-data={data.name}-clusters={clusters}--cluster_bits={number_cluster_bits}'
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

    iterations = len(clusters)
    for x in range(iterations):
        print('percentage of weights ', (1/iterations)*(x+1))
        print('no of clusters ', clusters[x])
        logger = TensorBoardLogger(
                save_dir = f'{os.getcwd()}/experiments/', 
                version = f'_iteration{x}', 
                name = experiment_name
                )
        trainer = pl.Trainer(gpus=1, max_epochs = 100, logger = logger, num_sanity_val_steps = 0)
        trainer.fit(model, data)
        trainer.test()
        model.percentage_fixed = (1/iterations)*(x+1)
        model.cluster_prune(clusters[x])
        model.reset_weights()
        model.print_the_number_of_unique_params()
        model.fixing_iteration +=1
    print('Final test')
    model.print_the_number_of_unique_params(True)
    model.print_the_number_of_unique_params()
    trainer.test()

run_experiment('set_1', model, cifar, clusters)
