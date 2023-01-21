
CLIP_VARIANT=RN50
DEBUG_TAG=debug

# python nat_policies/train_latent_dynamics.py train.task_group=put-block-in-bowl-seen-colors \
#         tag=${DEBUG_TAG} \
#         debug=True \
#         train.clip_variant=${CLIP_VARIANT} \
#         train.batch_size=128 \
#         train.log=False \
#         train.n_demos=1000 \
#         train.n_finetune_epochs=10 \
#         dataset.cache=True \
        #train.checkpoint=finetune/put-block-in-bowl-seen-colors-roboclip_RN50_FiLM/checkpoints/last.ckpt \

python nat_policies/train.py train.task=put-block-in-bowl-seen-colors \
        tag=debug \
        debug=True \
        train.agent=cliport_visual \
        #train.use_gt_goals=True \
        train.attn_stream_fusion_type=add \
        train.trans_stream_fusion_type=conv \
        train.goal_fusion_type=mult_ \
        train.n_demos=1000 \
        train.n_steps=201000 \
        train.exp_folder=exps \
        train.log=False \
        dataset.cache=True \
        #train.roboclip_ckpt_path=finetune/put-block-in-bowl-seen-colors-roboclip_RN50_FiLM/checkpoints/best-v1.ckpt


# TRAIN_TAG=roboclip-GT-Vis_RN50_FiLM_IL

# python nat_policies/eval.py eval_task=put-block-in-bowl-seen-colors \
#         model_task=put-block-in-bowl-seen-colors \
#         train_tag=${TRAIN_TAG} \
#         agent=roboclip \
#         mode=val \
#         n_demos=100 \
#         train_demos=1000 \
#         checkpoint_type=val_missing \
#         exp_folder=exps