import os
import math
import hydra
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from nat_policies.datasets import RavensFinetuneDataset
from nat_policies.agents.finetune_clip_agent import FinetuneCLIPAgent


@hydra.main(config_path="./cfg", config_name='finetune_clip')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(
        project=cfg['wandb']['logger']['project'], name=cfg['tag']
    ) if cfg['train']['log'] and not cfg['debug'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['finetune_dir'], 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    ) if not cfg['debug'] else None

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    LP_phase = cfg['train']['LP_phase']
    max_epochs = cfg['train']['n_finetune_epochs']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)
    batch_size = cfg['train']['batch_size']
    slurm_n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    num_workers = max(1, slurm_n_cpus - 1)

    print(f'Num workers for dataloaders: {num_workers}')

    # Handle FT phase
    LP_checkpoint_path = None
    should_load_ckpt = not LP_phase and cfg['train']['fusion_type'] != 'add'
    if should_load_ckpt:
        # Get LP checkpoint
        LP_run_dir = cfg['train']['LP_dir']
        LP_checkpoint_path = os.path.join(LP_run_dir, 'checkpoints', 'best.ckpt')
        assert os.path.exists(LP_checkpoint_path)

    # Datasets
    train_ds = RavensFinetuneDataset(
        os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_ds = RavensFinetuneDataset(
        os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False
    )
    val_batch_size = 10 if cfg['debug'] else len(val_ds)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=num_workers)

    # Trainer
    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=True,
        check_val_every_n_epoch=2,
        log_every_n_steps=1,
    )

    # Initialize agent
    agent = FinetuneCLIPAgent(name, cfg, train_dl, val_dl, model_ckpt_path=LP_checkpoint_path)

    # Main training loop
    trainer.fit(agent)


if __name__ == '__main__':
    main()




# def main(args):
#     # # Load clip model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = RoboCLIP._load_clip(device=device)
#     freeze_layers(model)

#     print(model)
#     print(f'Num trainable parameters (visual): {count_parameters(model.visual)}')
#     print(f'Num trainable parameters (language): {count_parameters(model.transformer)}')
#     print(f'Num trainable parameters (total): {count_parameters(model)}')

#     import pdb; pdb.set_trace()

#     # Load dataset
#     train_dataset = CLIPortDataset(
#         data_path='data/put-block-in-bowl-seen-colors-train', clip_preprocess=preprocess
#     )
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     val_dataset = CLIPortDataset(
#         data_path='data/put-block-in-bowl-seen-colors-val', clip_preprocess=preprocess
#     )
#     val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

#     if args.loss_fn == 'clip':
#         loss_img = nn.CrossEntropyLoss()
#         loss_text = nn.CrossEntropyLoss()
        
#     optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

#     N_train = len(train_dataset)
#     N_val = len(val_dataset)
#     debug = args.debug
#     save_dir = 'logs/experiment/'

#     if not debug:
#         wandb.init(project='nat_policies')
#         wandb.config = {
#             'loss_fn': args.loss_fn,
#             'learning_rate': args.lr,
#             'batch_size': args.batch_size
#         }
#         save_dir = os.path.join('logs', wandb.run.name)

#     Path(save_dir).mkdir(parents=True, exist_ok=True)

#     def train_loop(model, dataloader):
#         model.train()
#         avg_loss_train = 0
#         num_batches = 0
#         for i, batch in enumerate(dataloader):
#             optimizer.zero_grad()

#             start_imgs, lang_goals, goal_imgs = batch
#             start_imgs = start_imgs.to(device)
#             lang_goals = lang_goals.to(device)
#             goal_imgs = goal_imgs.to(device)
            
#             if args.loss_fn == 'L2':
#                 loss, embeddings_data = L2_loss(model, start_imgs, lang_goals, goal_imgs)
#             elif args.loss_fn == 'clip':
#                 loss, embeddings_data = constrastive_loss(
#                     model, start_imgs, lang_goals, goal_imgs,
#                     loss_img, loss_text
#                 )

#             if i == 0: # Only compute stats on the first batch for now
#                 stats = update_stats(embeddings_data, split='TRAIN')

#             loss.backward()
#             avg_loss_train += loss.detach()
#             num_batches += 1

#             #convert_models_to_fp32(model)
#             optimizer.step()
#             #clip.model.convert_weights(model)

#         avg_loss_train /= num_batches
#         stats.update({
#             'TRAIN_loss': avg_loss_train.item()
#         })
#         return stats

#     def val_loop(model, dataloader):
#         model.eval()
#         avg_loss_val = 0
#         num_batches = 0
#         with torch.no_grad():
#             for batch in dataloader:
#                 start_imgs, lang_goals, goal_imgs = batch
#                 start_imgs = start_imgs.to(device)
#                 lang_goals = lang_goals.to(device)
#                 goal_imgs = goal_imgs.to(device)
                
#                 if args.loss_fn == 'L2':
#                     loss, embeddings_data = L2_loss(model, start_imgs, lang_goals, goal_imgs)
#                 elif args.loss_fn == 'clip':
#                     loss, embeddings_data = constrastive_loss(
#                         model, start_imgs, lang_goals, goal_imgs,
#                         loss_img, loss_text
#                     )

#                 stats = update_stats(embeddings_data, split='VAL')
#                 avg_loss_val += loss
#                 num_batches += 1

#         avg_loss_val /= num_batches
#         stats.update({
#             'VAL_loss': avg_loss_val.item()
#         })
#         return stats
    
#     for epoch in range(args.num_epochs):
#         eval_epoch = (epoch % args.eval_freq == 0)
#         stats = train_loop(model, train_dataloader)

#         if eval_epoch:
#             val_stats = val_loop(model, val_dataloader)
#             stats.update(val_stats)
        
#         if epoch % args.log_freq == 0:
#             print(f'Epoch {epoch} | {stats}')
#             if not debug:
#                 wandb.log(stats, step=epoch)

#         if epoch % args.save_freq == 0 and not debug:
#             model_fp = os.path.join(save_dir, f'ckpt_epoch={epoch}.pt')
#             torch.save(model.state_dict(), model_fp)
