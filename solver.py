import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import load_svhn_dataset
from sklearn import metrics
import numpy as np 
from tqdm import tqdm
import torchvision
import pandas as pd
from IPython.display import display, clear_output
import os
import time 
import matplotlib.pyplot as plt 
import json
from torch.optim.lr_scheduler import CyclicLR

class Solver():
    def  __init__(self, model, criterion, optimizer, dl_train, dl_test, device, args):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.device = device
        self.args = args
        self.test_acc = 0
        self.best_test_acc = 0
        self.test_loss = np.inf
        self.train_loss = np.inf
        self.train_acc = 0
        self.df = pd.DataFrame(columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'time'])
        self.df.index.name='epochs'
        self.loss_list = []


    def save_state(self):
        # Save the initial weights and optimizer state
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        optimizer_state = {k: v for k, v in self.optimizer.state_dict().items()}
        return weights, optimizer_state


    def restore_weights(self, initial_model_weights, initial_optimizer_state):
        # Restore the initial weights of the model
        self.model.load_state_dict(initial_model_weights)

        # Reinitialize the optimizer with the original parameters
        new_optimizer = type(self.optimizer)(self.model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(initial_optimizer_state)
        self.optimizer = new_optimizer


    def train(self, lr = None):

        if lr is not None:
          self.optimizer.param_groups[0]['lr'] = lr

        scaler = torch.cuda.amp.GradScaler()

        iter_per_epoch = len(self.dl_train)

        self.create_directory()

        # Set up the cyclic learning rate scheduler
        #if cyclic_lr:
        #    scheduler = CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2000, mode='triangular')


        for epoch in range(self.args.epochs): 
            
            start_time = time.time()  # Record the start time of the epoch

            self.model.train()
            actuals = []
            preds = []
            outputs = []
            self.loss_list = []

            for i, (inputs, labels) in enumerate(tqdm(self.dl_train, 'Training')):
                
                self.optimizer.zero_grad()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(dtype=torch.float16):

                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                #if cyclic_lr:
                #    scheduler.step()

                actuals += labels.tolist()
                preds += logits.argmax(1).tolist()
                self.loss_list.append(loss.detach().cpu().numpy())

            self.train_loss = np.mean(self.loss_list)
            self.train_acc = metrics.accuracy_score(actuals, preds)

            self.test()
            
            end_time = time.time()
            epoch_time = end_time - start_time
              
            self.update_results(epoch, epoch_time)

            self.store_results()

            clear_output()
            display(self.df)


    def update_results(self, epoch, epoch_time):
      
        self.df.loc[(epoch + 1),:] = self.train_loss, self.test_loss, self.train_acc, self.test_acc, epoch_time
            

    def create_directory(self):
      
      if self.args.save_dir is not None: 
          os.makedirs(self.args.save_dir, exist_ok=True) 

          # Convert args to a dictionary
          args_dict = vars(self.args)

          # Save args to a JSON file
          with open(os.path.join(self.args.save_dir, "args.json"), "w") as json_file:
              json.dump(args_dict, json_file)

    def store_results(self):
        
        if self.args.save_dir is not None: 
            self.df.to_csv(os.path.join(self.args.save_dir, 'results.csv'))

            if self.test_acc > self.best_test_acc:
                torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth')) # save best model


    def test(self):
    
        self.model.eval()
        actuals = []
        preds = []
        outputs = []
        loss_list = []

        with torch.no_grad():

            for i, (inputs, labels) in enumerate(tqdm(self.dl_test, 'making predictions')):
        
                inputs, labels = inputs.to(self.device), labels.to(self.device)
      
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                
                actuals += labels.tolist()
                preds += logits.argmax(1).tolist()
                loss_list.append(loss.item())

            self.test_loss = np.mean(loss_list)
            self.test_acc = metrics.accuracy_score(actuals, preds)


    def test_mc_dropout(self, num_mc_samples=10):
        self.model.train()  # Set the model to training mode so dropout is active

        actuals = []
        preds_list = []
        outputs = []
        loss_list = []

        with torch.no_grad():

            for i, (inputs, labels) in enumerate(tqdm(self.dl_test, 'making predictions')):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Monte Carlo Dropout
                mc_preds = torch.zeros((num_mc_samples, labels.size(0), self.args.n_classes), device=self.device)
                for mc_sample in range(num_mc_samples):
                    logits = self.model(inputs)  # Dropout is active due to model.train()
                    mc_preds[mc_sample] = logits

                # Compute mean prediction
                mean_logits = mc_preds.mean(dim=0)

                # Compute loss
                loss = self.criterion(mean_logits, labels)

                actuals += labels.tolist()
                preds_list += mean_logits.argmax(1).tolist()
                loss_list.append(loss.item())

            self.test_loss = np.mean(loss_list)
            self.test_acc = metrics.accuracy_score(actuals, preds_list)  


    def find_lr(self, init_lr=1e-5, final_lr=0.1, n_steps=100, show_plot=True):
        
        # store the initial weights
        initial_model_weights, initial_optimizer_state = self.save_state()
        
        losses = []
        lrs = []
        lr_step = (final_lr / init_lr)**(1/(n_steps-1))
        current_lr = init_lr
        smoothing_window = int(n_steps/10)

        self.model.train()
        self.optimizer.param_groups[0]['lr'] = init_lr

        for step in tqdm(range(n_steps)):

            inputs, labels = next(iter(self.dl_train))
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            lrs.append(current_lr)
            losses.append(loss.item())

            current_lr *= lr_step
            self.optimizer.param_groups[0]['lr'] = current_lr

        results = pd.DataFrame({'loss': losses}, index=lrs)
        results.index.name = 'lrs'
        results = results.groupby(results.index).mean()

        smoothed_results = results.rolling(window=smoothing_window, center=True).mean()
        min_lr = smoothed_results['loss'].idxmin() 
        min_loss = smoothed_results['loss'].min()  

        if show_plot is True:
            # plt.plot(results.index, results['loss'], label='Loss')
            plt.plot(smoothed_results.index, smoothed_results['loss'], label='Loss_smoothed')

            plt.scatter(min_lr, min_loss, color='red', label='Min Loss', s=100)

            plt.xscale("log")
            plt.xlabel("Learning Rate (log scale)")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        # restore the original weights
        self.restore_weights(initial_model_weights, initial_optimizer_state)

        return min_lr / 10 


def define_parser():
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=3, help="Number of channels")
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--load_model', type=str, default='', help='path to a model checkpoint')

    parser.add_argument("--embed_dim", type=int, default=96, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2048, help="forward multiplier")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")
    parser.add_argument("--save_dir", type=str, default=None, help="define a directory where the results are saved")

    return parser 