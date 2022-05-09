# Cost-Free Incremental Learning
Run training with:
```
PYTHONPATH=$(pwd) python cf_il/main.py \
    --dataset 'seq-cifar10' \
    --lr 0.001 \
    --momentum 0.9 \
    --n_epochs 1 \
    --batch_size 10 \
    --minibatch_size 10 \
    --buffer_size 100 \
    --alpha 0.2 \
    --tensorboard
```