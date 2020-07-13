import torch
import torch.nn as nn
import numpy as np
import time
import pylab as pl
from IPython import display

class LSTM(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
    super().__init__()
    self.hidden_layer_size=hidden_layer_size
    self.lstm=nn.LSTM(input_size,hidden_layer_size)

    self.linear=nn.Linear(hidden_layer_size,output_size)


    self.hidden_cell=(torch.zeros(1,1,self.hidden_layer_size),
                      torch.zeros(1,1,hidden_layer_size))
    self.loss_array = []

  def forward(self, input_seq):
    lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1),
                                           self.hidden_cell)
    predictions = self.linear(lstm_out.view(len(input_seq),-1))

    return predictions[-1]
  def test(self,train_inout_seq,loss_function,device):
    loss=[]
    preds=[]
    for seq,  _, labels in train_inout_seq:
      with torch.no_grad():
        self.hidden_cell=(torch.zeros(1,1,self.hidden_layer_size).to(device),
                         torch.zeros(1,1,self.hidden_layer_size).to(device))
        y_pred = self.forward(seq)
        preds.append(y_pred.item())
        y_pred=y_pred.view(1,-1).to(device)
        single_loss=loss_function(y_pred,labels)
        loss.append(single_loss.item())
    return loss, preds
  def train(self, train_inout_seq, optimizer, loss_function, device, epochs=10, lr=1e-5,draw_fig=False):
    for i in range(epochs):
      loss = []
      for seq, _,labels in train_inout_seq:
        optimizer.zero_grad()
        self.hidden_cell=(torch.zeros(1,1,self.hidden_layer_size).to(device),
                         torch.zeros(1,1,self.hidden_layer_size).to(device))
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

class LSTM_softmax(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=10, output_size=10):
    super().__init__()
    self.hidden_layer_size=hidden_layer_size
    self.lstm=nn.LSTM(input_size,hidden_layer_size)

    self.linear=nn.Linear(hidden_layer_size,output_size)

    self.softmax=nn.Softmax()

    self.hidden_cell=(torch.zeros(1,1,self.hidden_layer_size),
                      torch.zeros(1,1,hidden_layer_size))
  def forward(self, input_seq):
    lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1),
                                           self.hidden_cell)
    predictions = self.linear(lstm_out.view(len(input_seq),-1))

    softmax = self.softmax(predictions[-1])

    return softmax

