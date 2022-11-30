# nat_policies

### Replicating Results in the Report

To finetune a CLIP model on data from a single task:

```bash
python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
                        tag=finetune_CLIP \
                        train.log=True \
                        train.n_demos=1000 \
                        train.n_finetune_epochs=500 \
                        dataset.cache=True

```


To evaluate all the checkpoints on the validation set and record the best-performing checkpoint:

```bash
python nat_policies/eval.py eval_task=put-block-in-bowl-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps 
```

To run the best-performing checkpoint on the test dataset:

```bash
python nat_policies/eval.py eval_task=put-block-in-bowl-seen-colors \
                        agent=cliport \
                        mode=test \
                        n_demos=100 \
                        train_demos=1000 \
                        checkpoint_type=test_best \
                        exp_folder=exps
```
