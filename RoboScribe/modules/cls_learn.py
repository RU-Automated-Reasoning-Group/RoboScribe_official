import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

import pdb

class PredicateCls(nn.Module):
    def __init__(self, in_dim, layer_dims=[128, 128]):
        super(PredicateCls, self).__init__()
        # init parameters
        self.layer_dims = layer_dims
        self.layers = []
        # init linear layers
        for inp, outp in zip([in_dim]+layer_dims[:-1], layer_dims[1:]+[1]):
            self.layers += [nn.Linear(inp, outp), 
                            nn.ReLU()]
        self.layers.pop(-1)
        self.layers = nn.Sequential(*self.layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        output = self.layers(obs)

        return output

    def execute(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        output = self.layers(obs)
        output = (self.sigmoid(output) > 0.5).int().item()

        return output
    
    def pretty(self):
        return ''

class ClsLearner:
    def __init__(self, cls, train_epoch=100, batch_size=128, learning_rate=1e-4, use_cuda=True):
        self.cls = cls
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cls.cuda()

    def do_learn(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        # upsample
        positive_idx = torch.nonzero(y==1).squeeze(1)
        positive_data = X[positive_idx]
        X = torch.concat([X, positive_data.repeat(24, 1)], dim=0)
        y = torch.concat([y, y[positive_idx].repeat(24)], dim=0)

        d_ids = np.arange(X.shape[0]).tolist()
        act_fun = nn.Sigmoid()
        loss_fun = nn.BCELoss(reduction='none')
        w1 = 1.0
        w2 = 1.0

        optim = torch.optim.Adam(self.cls.parameters(), lr=self.learning_rate)

        # train for certain epochs
        for epoch in tqdm(range(self.train_epoch)):
            # train
            random.shuffle(d_ids)
            self.cls.train()
            for d_id in range(0, X.shape[0], self.batch_size):
                # init data
                X_batch = X[d_ids[d_id: d_id+self.batch_size]]
                y_batch = y[d_ids[d_id: d_id+self.batch_size]]
                if self.use_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # get loss
                pred_y = self.cls(X_batch)

                loss = loss_fun(act_fun(pred_y).squeeze(1), y_batch)
                weight = w1 * (1-y_batch) + w2 * y_batch
                loss = torch.mean(weight * loss)

                # backprop
                optim.zero_grad()
                loss.backward()
                optim.step()

            # evaluate
            if (epoch+1) % 20 == 0:
                self.cls.eval()
                correct_num = 0
                pos_correct_num = 0
                neg_correct_num = 0
                with torch.no_grad():
                    for d_id in range(0, X.shape[0], self.batch_size):
                        # init data
                        X_batch = X[d_id: d_id+self.batch_size]
                        y_batch = y[d_id: d_id+self.batch_size]
                        if self.use_cuda:
                            X_batch = X_batch.cuda()
                            y_batch = y_batch.cuda()
                        
                        # get loss
                        pred_y = self.cls(X_batch)
                        pred_label = (act_fun(pred_y) > 0.5).int().squeeze(1)

                        correct_num += torch.sum(pred_label == y_batch).item()
                        pos_correct_num += torch.sum((pred_label==y_batch).float() * (y_batch==1).float()).item()
                        neg_correct_num += torch.sum((pred_label==y_batch).float() * (y_batch==0).float()).item()

                    print('for epoch {} get evaluation accuracy: {}  positive accuracy: {}  negative accuracy: {}'\
                        .format(epoch, float(correct_num)/X.shape[0], float(pos_correct_num)/torch.sum(y).item(), float(neg_correct_num)/torch.sum(1-y).item()))

        return self.cls