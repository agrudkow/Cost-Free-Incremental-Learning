# Cost-Free Incremental Learning
The aim of our work was to implement the solution for Incremental Learning (IL) named [Cost-Free Incremental Learning](https://arxiv.org/abs/2103.12216). The task is to perform classification given images in the consecutive tasks where only the subset of image classes is present. The solution (called later on CF-IL-MRP for brevity) is inspired by replay paradigm. It consists of two mechanisms (shown in Figure 1): Cost-Free Incremental Learning (CF-IL) and Memory Recovery Paradigm (MRP). Our implementation is based on the [original implementation](https://github.com/MozhganPourKeshavarz/Cost-Free-Incremental-Learning) which at the time of our implementation lacked notable amount of codebase.

Implementation of the Cost-Free Incremental Learning is placed in directory `cf_il`. The rest of the folder comes from original implementation (we use/share some parts of code).

## Environment setup
```
conda create -n CF IL MRP python=3.8 -y
conda activate CF IL MRP

# Install required production packages
pip install -r requirements.txt
# or development packages
pip install -r requirements-dev.txt
```
## Configuration
Config file is in `cf_il/conf/config.yaml`.

Parameters:
* dataset (str) - dataset name
* tensorboard (bool) - log to tensorboard
* validation (bool) - use validation dataset
* n_epochs (int) - number of epochs for one task
* lr (float) - learning rate for main optimizer
* momentum (float) - momentum for main optimizer
* batch_size (int) - batch size
* buffer_size (int) - buffer size for data impressions
* buffer_batch_size (int) - batch size for data impressions
* alpha (float) - weight of the loss for the data impressions
* eta (float) - parameter controlling how different sampled logit cant be from original class representation vector
* tau (float - Cross Entropy temperature
* scale (Tuple[float, float]) - tuple of scales to be used during computing center of the mass for the Dirichlet distribution. First element will be used for the first half of the classes and second for the rest
* synth_img_optim_steps (int) - number of saved data impressions. Data impressions are randomly selected
* synth_img_optim_lr (float) - maximal amount of iterations of sampling for one logit
* synth_img_save_dir (str) - dir for saving generated data impressions. If `None` data impressions are not saved
* synth_img_save_num (int) - number of saved data impressions. Data impressions are randomly selected
* dirichlet_max_iter ([int) - maximal amount of iterations of sampling for one logit
## Training
Run training with:
```
PYTHONPATH=$(pwd) python cf_il/main.py
```

### Colab
In `notebooks/cfil_colab.ipynb` there is a Colab notebook which allow to train model on Colab servers.
