"""Main training script."""

import os
import hydra
import torch

from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import cliport.agents as agents

from nat_policies.agents import (
    CLIPortVisualGoalAgent, RoboCLIPAgent
)
from nat_policies.datasets import RavensDataset, RavensMultiTaskDataset


@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(
        project=cfg['wandb']['logger']['project'], name=cfg['tag']
    ) if (cfg['train']['log'] and not cfg['debug']) else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        fast_dev_run=False, #cfg['debug'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max_epochs // 50,
        resume_from_checkpoint=last_checkpoint,
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    use_visual_goals = cfg['train']['use_gt_goals']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RavensDataset(
            os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True,
        )
        val_ds = RavensDataset(
            os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False,
        )
    
    # Initialize agent
    if agent_type == 'cliport_visual':
        agent = CLIPortVisualGoalAgent(name, cfg, train_ds, val_ds)
    elif agent_type == 'roboclip':
        agent = RoboCLIPAgent(name, cfg, train_ds, val_ds)
    elif agent_type in agents.names:
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
    else:
        raise Exception()

    # Main training loop
    trainer.fit(agent)

if __name__ == '__main__':
    main()