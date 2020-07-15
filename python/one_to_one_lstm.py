import torch
import torch.nn as nn
import numpy as np
import time
import pylab as pl
from IPython import display
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

class LSTM(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=10, output_size=1,num_layers=1,use_softmax=False):
    super().__init__()
    self.hidden_layer_size=hidden_layer_size
    self.num_layers=num_layers
    self.lstm=nn.LSTM(input_size,hidden_layer_size,num_layers)

    self.linear=nn.Linear(hidden_layer_size,output_size)


    self.hidden_cell=(torch.zeros(num_layers,1,self.hidden_layer_size),
                      torch.zeros(num_layers,1,hidden_layer_size))
    self.use_softmax=use_softmax
    if use_softmax: self.softmax=nn.Softmax()

    self.loss_array = []

  def forward(self, input_seq):
    lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1),
                                           self.hidden_cell)
    predictions = self.linear(lstm_out.view(len(input_seq),-1))

    return predictions[-1]
  def test(self,train_inout_seq,loss_function,device):
    loss=[]
    preds=[]
    for seq, labels in train_inout_seq:
      with torch.no_grad():
        self.hidden_cell=(torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device),
                         torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device))
        y_pred = self.forward(seq)
        preds.append(y_pred.item())
        y_pred=y_pred.view(1,-1).to(device)
        single_loss=loss_function(y_pred,labels)
        loss.append(single_loss.item())
    return loss, preds
  def train(self, train_inout_seq, optimizer, loss_function, device, epochs=10, lr=1e-5,draw_fig=False):
    for i in range(epochs):
      loss = []
      for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        self.hidden_cell=(torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device),
                         torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device))
        y_pred = self.forward(seq)
        y_pred=y_pred.view(1,-1).to(device)
        single_loss=loss_function(y_pred,labels)
        single_loss.backward()
        loss.append(single_loss.item())
        optimizer.step()

      self.loss_array.append(np.average(loss))
      if draw_fig:
        pl.clf()
        pl.plot(self.loss_array)
        display.clear_output(wait=True)
        display.display(pl.gcf())
      if i%1==0: print(f'epoch: {i:3} loss: {np.average(loss):10.8f} lr: {lr:10.8f}')
    return self.loss_array

class BNN_LSTM(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=10, output_size=1,num_layers=1,use_softmax=False):
    super().__init__()
    self.hidden_layer_size=hidden_layer_size
    self.num_layers=num_layers
    self.lstm=nn.LSTM(input_size,hidden_layer_size,num_layers)

    self.blinear=BayesianLinear(hidden_layer_size,output_size)

    self.hidden_cell=(torch.zeros(num_layers,1,self.hidden_layer_size),
                      torch.zeros(num_layers,1,hidden_layer_size))
    self.use_softmax=use_softmax
    if use_softmax: self.softmax=nn.Softmax()
    
    self.loss_array = []

  def forward(self, input_seq):
    lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1),
                                           self.hidden_cell)
    predictions = self.blinear(lstm_out.view(len(input_seq),-1))

    return predictions[-1]
  def test(self,train_inout_seq,loss_function,device):
    loss=[]
    preds=[]
    for seq, labels in train_inout_seq:
      with torch.no_grad():
        self.hidden_cell=(torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device),
                         torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device))
        y_pred = self.forward(seq)
        preds.append(y_pred.item())
        y_pred=y_pred.view(1,-1).to(device)
        single_loss=loss_function(y_pred,labels)
        loss.append(single_loss.item())
    return loss, preds
  def train(self, train_inout_seq, optimizer, loss_function, device, epochs=10, lr=1e-5,draw_fig=False):
    for i in range(epochs):
      losses = []
      for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        self.hidden_cell=(torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device),
                         torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device))
        y_pred = self.forward(seq)
        y_pred=y_pred.view(1,-1).to(device)
        single_loss=loss_function(y_pred,labels)
        complexity_loss = self.nn_kl_divergence()
        loss = single_loss + complexity_loss
        loss.backward()

        losses.append(single_loss.item())
        optimizer.step()

      self.loss_array.append(np.average(loss))
      if draw_fig:
        pl.clf()
        pl.plot(self.loss_array)
        display.clear_output(wait=True)
        display.display(pl.gcf())
      if i%1==0: print(f'epoch: {i:3} loss: {np.average(loss):10.8f} lr: {lr:10.8f}')
    return self.loss_array
def evaluate_regression(self,train_inout_seq,samples = 100, std_multiplier = 2):
  preds = []
  for seq, labels in train_inout_seq:
    preds.extend([self.forward(seq) for i in range(samples)])
  preds = torch.stack(preds)
  means = preds.mean(axis=0)
  stds = preds.std(axis=0)
  ci_upper = means + (std_multiplier * stds)
  ci_lower = means - (std_multiplier * stds)
  ic_acc = (ci_lower <= labels) * (ci_upper >= labels)
  ic_acc = ic_acc.float().mean()
  return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()