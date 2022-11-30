import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from pytorch_lightning import LightningModule

from cliport.utils import utils
from cliport.models.core.clip import tokenize
from nat_policies.models.core import RoboCLIP
from nat_policies.utils.eval_utils import ground_truth_L2, cross_batch_L2, knn_classification, start_pred_goal_ratio
from nat_policies.utils.common import count_parameters


class FinetuneCLIPAgent(LightningModule):
    def __init__(self, name, cfg, train_dataloader, val_dataloader, model_ckpt_path=None):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0

        self.LP_phase = cfg['train']['LP_phase']
        self.roboclip = RoboCLIP(
            clip_variant=cfg['train']['clip_variant'],
            LP_phase=self.LP_phase,
            device=self.device_type
        )
        n_trainable_before = count_parameters(self.roboclip, count_trainable_only=True)
        if model_ckpt_path is not None:
            print(f'Loading model ckpt from: {model_ckpt_path}')
            ckpt = torch.load(model_ckpt_path)
            model_state_dict = ckpt['state_dict']
            self.load_state_dict(model_state_dict)

        n_trainable_after = count_parameters(self.roboclip, count_trainable_only=True)
        assert n_trainable_before == n_trainable_after

        self.ce_loss_vis = nn.CrossEntropyLoss()
        self.ce_loss_fusion = nn.CrossEntropyLoss()

        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))
        print(f'Num trainable parameters: {n_trainable_after}')
        print(f'LP Phase: {self.LP_phase}')

    '''Loss computation methods'''

    def contrastive_loss(self, start_imgs, lang_goals, goal_imgs, return_emb=False):
        # Compute the predicted goal image embedding
        pred_goal_embeddings, start_embeddings, lang_embeddings = self.roboclip(start_imgs, lang_goals)
        
        # Compute the actual goal image embedding
        goal_embeddings = self.roboclip.encode_image(goal_imgs)
        
        start_embeddings_normalized = start_embeddings / start_embeddings.norm(dim=1, keepdim=True)
        lang_embeddings_normalized = lang_embeddings / lang_embeddings.norm(dim=1, keepdim=True)
        pred_goal_embeddings_normalized = pred_goal_embeddings / pred_goal_embeddings.norm(dim=1, keepdim=True)
        goal_embeddings_normalized = goal_embeddings / goal_embeddings.norm(dim=1, keepdim=True)

        # Compute similarity matrix for (start_img+lang_goal, goal_img) pairs
        logit_scale = self.roboclip.logit_scale.exp()
        logits_per_pred_goal = logit_scale * pred_goal_embeddings_normalized @ goal_embeddings_normalized.t()
        logits_per_goal = logits_per_pred_goal.t()

        ground_truth = torch.arange(len(logits_per_goal), dtype=torch.long, device=logits_per_goal.device)
        goal_similarity_loss = (
            self.ce_loss_vis(logits_per_pred_goal, ground_truth) + 
            self.ce_loss_vis(logits_per_goal, ground_truth)
        ) / 2
        loss = goal_similarity_loss

        # if self.finetune_clip_layers:
        #     # # Compute an extra similarity matrix just for language pairs (to prevent degenerate solution)
        #     # logits_per_text = logit_scale * lang_embeddings_normalized @ lang_embeddings_normalized.t()
        #     # lang_similarity_loss = self.ce_loss_lang(logits_per_text, ground_truth)
        #     # loss += lang_similarity_loss

        emb_data = None
        if return_emb:
            emb_data = {
                'start_embeddings_normalized': start_embeddings_normalized,
                'lang_embeddings_normalized': lang_embeddings_normalized,
                'pred_goal_embeddings_normalized': pred_goal_embeddings_normalized,
                'goal_embeddings_normalized': goal_embeddings_normalized,
                'start_embeddings_unnormalized': start_embeddings,
                'goal_embeddings_unnormalized': goal_embeddings,
            }

        return loss, emb_data

    '''Utility methods'''

    def preprocess_batch(self, batch):
        start_imgs, lang_goals, goal_imgs = batch['start_img'], batch['lang_goal'], batch['goal_img']
        lang_goals = tokenize(lang_goals).to(self.device_type)
        return start_imgs, lang_goals, goal_imgs

    '''Overridden methods for PyTorch Lightning'''
    
    def configure_optimizers(self):
        if self.LP_phase:
            optimizer = torch.optim.Adam(
                self.roboclip.parameters(),
                lr=1e-4, betas=(0.9, 0.98), eps=1e-6
            )
        else:
            optimizer = torch.optim.Adam(
                self.roboclip.parameters(),
                lr=1e-5, betas=(0.9, 0.98), eps=1e-6
            )

        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
    
    def training_step(self, batch, batch_idx):
        self.roboclip.train()

        start_imgs, lang_goals, goal_imgs = self.preprocess_batch(batch)
        loss, _ = self.contrastive_loss(start_imgs, lang_goals, goal_imgs, return_emb=False)

        self.log('train/loss', loss.detach().item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.roboclip.eval()

        start_imgs, lang_goals, goal_imgs = self.preprocess_batch(batch)
        with torch.no_grad():
            loss, emb_data = self.contrastive_loss(start_imgs, lang_goals, goal_imgs, return_emb=True)
            start_embeddings, lang_embeddings, pred_goal_embeddings, goal_embeddings = (
                emb_data['start_embeddings_normalized'],
                emb_data['lang_embeddings_normalized'],
                emb_data['pred_goal_embeddings_normalized'],
                emb_data['goal_embeddings_normalized']
            )
            start_embeddings_unnormalized = emb_data['start_embeddings_unnormalized']
            goal_embeddings_unnormalized = emb_data['goal_embeddings_unnormalized']

            pred_goal_real_goal_dist = ground_truth_L2(pred_goal_embeddings, goal_embeddings)
            pred_goal_start_img_dist = ground_truth_L2(pred_goal_embeddings, start_embeddings)
            start_img_goal_img_dist = ground_truth_L2(start_embeddings, goal_embeddings)
            cross_batch_goal_dist = cross_batch_L2(pred_goal_embeddings, goal_embeddings)
            cross_batch_start_img_dist = cross_batch_L2(start_embeddings, start_embeddings)
            cross_batch_lang_dist = cross_batch_L2(lang_embeddings, lang_embeddings)
            top_1_acc, top_5_acc = knn_classification(pred_goal_embeddings, goal_embeddings, K=5)
        
        self.log('val/loss', loss.detach().item())
        self.log('val/pred_goal_2_goal_img_dist', pred_goal_real_goal_dist)
        self.log('val/pred_goal_2_start_img_dist', pred_goal_start_img_dist)
        self.log('val/start_img_2_goal_img_dist', start_img_goal_img_dist)
        self.log('val/cross_batch_goal_dist', cross_batch_goal_dist)
        self.log('val/top_1_acc', top_1_acc)
        self.log('val/top_5_acc', top_5_acc)
        
        self.log('val/cross_batch_start_img_dist', cross_batch_start_img_dist)
        self.log('val/cross_batch_lang_dist', cross_batch_lang_dist)

        print(f'Val loss: {loss}')

        return dict(
            val_loss=loss
        )