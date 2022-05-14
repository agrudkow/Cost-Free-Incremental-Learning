# Based on training.py file

from argparse import Namespace
from typing import List

import numpy as np

from cf_il.model import CFIL
from datasets.utils.continual_dataset import ContinualDataset
from training import evaluate
from utils.loggers import print_mean_accuracy
from utils.status import progress_bar, create_stash
from utils.tb_logger import TensorboardLogger



import wandb  ####### PT



def train(
    model: CFIL,
    dataset: ContinualDataset,
    args: Namespace, 
    #wandb_writer: Namespace,
) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None

    # TODO: change to some resonable restriction
    # Currently model is recovering `num_images_per_class` images for each class
    # where it is set as args.minibatch_size below.
    assert args.buffer_size >= args.minibatch_size * dataset.N_CLASSES_PER_TASK

    model.net.to(model.device)

    num_tasks: int = dataset.N_TASKS
    num_classes_per_task: int = dataset.N_CLASSES_PER_TASK
    results: List[List[float]] = []
    results_mask_classes: List[List[float]] = []

    model_stash = create_stash(model, args, dataset)

    if args.tensorboard:
        #tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        #print(tb_logger.get_name())
        #model_stash['tensorboard_name'] = tb_logger.get_name()
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        

    for task_idx in range(num_tasks):
        model.net.train()
        train_loader, _ = dataset.get_data_loaders()

        if task_idx > 0:
            # Recover past experience
            current_num_of_trained_classes = task_idx * num_classes_per_task
            model.recover_memory(
                num_classes=current_num_of_trained_classes,
                num_images_per_class=args.minibatch_size,  # TODO: change to some resonable parameter
            )

  
            #img=model.buffer.get_data(1)[0][0]
            
            #print("img", np.moveaxis(img, 0, -1).shape)
            writer.add_image(f'synthetic_{task_idx}', model.buffer.get_data(1)[0][0])

            # Eval current model             ######### chyba niepotrzebne
            #accs = evaluate(model, dataset, last=True)           ######### po co tutaj jest last?
            #results[task_idx - 1] = results[task_idx - 1] + accs[0]
            #if dataset.SETTING == 'class-il':
            #    print("masked1", results_mask_classes)
            #    results_mask_classes[task_idx - 1] = results_mask_classes[task_idx - 1] + accs[1]
            #    print("masked2", results_mask_classes)
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, task_idx, loss)

                if args.tensorboard:
                    #tb_logger.log_loss(loss, args, epoch, task_idx, i)
                    writer.add_scalar("Train/loss", loss, task_idx, i)

                #if i>3:
                #    break

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = task_idx + 1
        model_stash['epoch_idx'] = 0

        # Finish training
        accs = evaluate(model, dataset)  ######### czy tutaj nie brakuje klasy? do sprawdzenia
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        print("ressssssssssssssssssss", results)
        print("ressssssssssssssssssss", results_mask_classes)
        #mean_acc = np.mean(np.array(results))
        #mean_acc_masked = np.mean(np.array(results_mask_classes))
        #print_mean_accuracy(mean_acc, task_idx + 1, dataset.SETTING)

        #model_stash['mean_accs'].append(mean_acc)

        #metrics_dict={}
        #for i in range(num_tasks):
        #    metrics_dict[i]=results

        res=results[-1]+[0] * (num_tasks - len(results[-1]))
        metrics_dict={}
        for i in range(num_tasks):
            metrics_dict[f"task_{i}"]=res[i]
        #print("dict", metrics_dict)
        #print("mean", np.mean(results[-1]))

        if args.tensorboard:
            #tb_logger.log_accuracy(np.array(accs), mean_acc, args, task_idx)
            writer.add_scalar("Train/Epoch loss", loss, task_idx)  #-- skąd wziąć loss?
            writer.add_scalars("Train/Epoch acc", metrics_dict, task_idx)
            writer.add_scalar("Train/Epoch acc_masked_avg", np.mean(results_mask_classes[-1]), task_idx)
            writer.add_scalar("Train/Epoch acc_avg", np.mean(results[-1]), task_idx)
            


