import argparse
from tracemalloc import start

import torch
import torch.nn as nn
import numpy as np
import clip
import wandb

from torch.utils.data import DataLoader
from torch.optim import Adam

from nat_policies.datasets import CLIPortDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--eval_freq', default=10)
    parser.add_argument('--log_freq', default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def freeze_layers(model):
    allowed_layers = [
        'visual.transformer.resblocks.11', 'visual.ln_post', # last attention block for visual encoder
        'transformer.resblocks.11', # layer attention block for text encoder
        'token_embedding', 'ln_final'
    ]
    for name, param in model.named_parameters():
        if any([name.startswith(layer) for layer in allowed_layers]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def L2Loss(model, start_imgs, lang_goals, goal_imgs):
    start_img_features = model.encode_image(start_imgs)
    lang_goal_features = model.encode_text(lang_goals.squeeze())

    # normalized features
    start_img_features = start_img_features / start_img_features.norm(dim=1, keepdim=True)
    lang_goal_features = lang_goal_features / lang_goal_features.norm(dim=1, keepdim=True)

    with torch.no_grad():
        goal_img_features = model.encode_image(goal_imgs)
        goal_img_features = goal_img_features / goal_img_features.norm(dim=1, keepdim=True)

    l2norm = torch.linalg.norm(start_img_features + lang_goal_features - goal_img_features, ord=2, dim=-1)
    loss = torch.mean(l2norm)
    return loss


def main(args):
    # Load clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    freeze_layers(model)
    print(f'Num parameters: {count_parameters(model)}')

    # Load dataset
    train_dataset = CLIPortDataset(
        data_path='data/put-block-in-bowl-seen-colors-train', clip_preprocess=preprocess
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CLIPortDataset(
        data_path='data/put-block-in-bowl-seen-colors-train', clip_preprocess=preprocess
    )
    val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = L2Loss
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    N = len(train_dataset)
    debug = args.debug

    if not debug:
        wandb.init(project='nat_policies')
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size
        }

    def train_loop(model, dataloader):
        model.train()
        avg_loss_train = 0
        for batch in dataloader:
            optimizer.zero_grad()

            start_imgs, lang_goals, goal_imgs = batch
            start_imgs = start_imgs.to(device)
            goal_imgs = goal_imgs.to(device)
            
            loss = loss_fn(model, start_imgs, lang_goals, goal_imgs)
            loss.backward()
            
            avg_loss_train += loss

            if device == "cpu":
                optimizer.step()
            else : 
                #convert_models_to_fp32(model)
                optimizer.step()
                #clip.model.convert_weights(model)

        avg_loss_train /= N
        stats = {
            'train_loss': avg_loss_train.detach().item()
        }
        return stats

    def val_loop(model, dataloader):
        model.eval()
        avg_loss_val = 0
        with torch.no_grad():
            for batch in dataloader:
                start_imgs, lang_goals, goal_imgs = batch
                start_imgs = start_imgs.to(device)
                goal_imgs = goal_imgs.to(device)
                
                loss = loss_fn(model, start_imgs, lang_goals, goal_imgs)
                avg_loss_val += loss

        avg_loss_val /= N
        stats = {
            'val_loss': avg_loss_val.detach().item()
        }
        return stats
    
    for epoch in range(args.num_epochs):
        stats = train_loop(model, train_dataloader)

        if epoch % args.eval_freq == 0:
            val_stats = val_loop(model, val_dataloader)
            stats.update(val_stats)
        
        if epoch % args.log_freq == 0:
            if not debug:
                wandb.log(stats, step=epoch)
                print(f'Epoch {epoch} | {stats}')


if __name__ == '__main__':
    main(get_args())