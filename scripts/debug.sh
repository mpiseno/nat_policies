
DEBUG_TAG=debug
CLIP_VARIANT=ViT

python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
        tag=${DEBUG_TAG} \
        debug=True \
        train.fusion_type=add \
        train.LP_phase=False \
        train.clip_variant=${CLIP_VARIANT} \
        train.log=False \
        train.n_demos=1000 \
        train.n_finetune_epochs=501 \
        dataset.cache=True \