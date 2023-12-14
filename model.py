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

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

    def forward(self, x):
        m, s, e = x.shape

        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
        xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
        xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE

        xq = xq.reshape([-1, s, self.head_embed_dim])  # B, H, Q, HE -> (BH), Q, HE
        xk = xk.reshape([-1, s, self.head_embed_dim])  # B, H, K, HE -> (BH), K, HE
        xv = xv.reshape([-1, s, self.head_embed_dim])  # B, H, V, HE -> (BH), V, HE

        xk = xk.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K
        x_attention = xq.bmm(xk)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(xv)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  # (BH), Q, HE -> B, H, Q, HE
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x


class MLP(nn.Module):
  def __init__(self, args):
      super().__init__()
      self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
      self.activation = nn.GELU()
      self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
  
      self.dropout1 = nn.Dropout(args.dropout)
      self.dropout2 = nn.Dropout(args.dropout)


  def forward(self, x):
      x = self.dropout1(self.fc1(x))
      x = self.activation(x)
      x = self.dropout2(self.fc2(x))

      return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.mlp = MLP(args)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x



class VisionTransformerPytorch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=args.n_attention_heads, dim_feedforward=(args.forward_mul*args.embed_dim), dropout=args.dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.n_layers)
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ImageDataset(Dataset):
    
    def __init__(self, array_images, labels, transforms):
        self.array_images = array_images
        self.transform = transforms
        self.labels = labels
        
    def __len__(self):
        return len(self.array_images)
    
    def __getitem__(self, idx):
        return self.transform(self.array_images[idx]), self.labels[idx]


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
        self.df = pd.DataFrame(columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        self.df.index.name='epochs'
        self.loss_list = []

    def train(self):
    
        iter_per_epoch = len(self.dl_train)

        for epoch in range(self.args.epochs): 
            
            self.model.train()
            actuals = []
            preds = []
            outputs = []
            self.loss_list = []

            for i, (imgs, labels) in enumerate(tqdm(self.dl_train, 'Training')):
        
                imgs, labels = imgs.to(self.device), labels.to(self.device)
        
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                actuals += labels.tolist()
                preds += logits.argmax(1).tolist()
                self.loss_list.append(loss.detach().cpu().numpy())

            self.train_loss = np.mean(self.loss_list)
            self.train_acc = metrics.accuracy_score(actuals, preds)

            self.test()
            
            self.update_results(epoch)

            self.store_results()

            clear_output()
            display(self.df)


    def update_results(self, epoch):
      
        self.df.loc[(epoch + 1),:] = self.train_loss, self.test_loss, self.train_acc, self.test_acc
            

    def store_results(self):
        
        if self.args.save_dir is not None: 
            os.makedirs(self.args.save_dir, exist_ok=True) 
            self.df.to_csv(os.path.join(self.args.save_dir, 'results.csv'))

            if self.test_acc > self.best_test_acc:
                torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth')) # save best model


    def test(self):
    
        self.model.eval()
        actuals = []
        preds = []
        outputs = []
        loss_list = []

        for i, (imgs, labels) in enumerate(tqdm(self.dl_test, 'making predictions')):
    
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            with torch.no_grad():

                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                
                actuals += labels.tolist()
                preds += logits.argmax(1).tolist()
                loss_list.append(loss.detach().cpu().numpy())

        self.test_loss = np.mean(loss_list)
        self.test_acc = metrics.accuracy_score(actuals, preds)
    

def get_loader(args, transform):

    train_images, train_labels, test_images, test_labels = load_svhn_dataset(args.train_path, args.test_path)

    ds_train = ImageDataset(train_images, train_labels, transform['train'])
    ds_test = ImageDataset(test_images, test_labels, transform['inference'])

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    return dl_train, dl_test


def define_parser():
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--log_step', type=int, default=50)
    
    parser.add_argument('--dset', type=str, default='mnist', help=['mnist', 'fmnist'])
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=3, help="Number of channels")
    parser.add_argument('--train_path', type=str, default='./data/')
    parser.add_argument('--test_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')
    
    parser.add_argument("--embed_dim", type=int, default=96, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2048, help="forward multiplier")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")
    parser.add_argument("--save_dir", type=str, default=None, help="define a directory where the model is saved")
    

    return parser