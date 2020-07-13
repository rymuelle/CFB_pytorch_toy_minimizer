{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "import one_to_one_lstm\n",
    "import time\n",
    "import pylab as pl\n",
    "from IPython import display"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 505,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<module 'one_to_one_lstm' from '/Users/ryandmueller/Documents/cfb_pytorch/CFB_pytorch_toy_minimizer/one_to_one_lstm.py'>"
      ]
     },
     "execution_count": 505,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import importlib\n",
    "importlib.reload(one_to_one_lstm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "dtype = torch.float\n",
    "device = torch.device(\"cpu\")\n",
    "#device = torch.device(\"cuda:0\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 578,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 500\n",
    "length_sample = 20\n",
    "label_std = 14\n",
    "seq_std = 14\n",
    "label_offset = 0\n",
    "label_std = label_std/40\n",
    "seq_std = seq_std/40\n",
    "label_offset =.5\n",
    "def create_sample(n_samples,length_sample,label_std,seq_std, label_offset):\n",
    "    labels = np.random.normal(label_offset, label_std, size=(n_samples, 1))\n",
    "    \n",
    "    len_seq_half = int(length_sample/2)\n",
    "    std_1=seq_std\n",
    "    std_2=seq_std*2\n",
    " \n",
    "    seqs = np.random.normal(0, std_1, size=(n_samples, len_seq_half)) \n",
    "    seqs2=np.random.normal(0, std_2, size=(n_samples, len_seq_half))\n",
    "    seqs=np.append(seqs,seqs2,axis=1)\n",
    "    seqs = (seqs+labels) \n",
    "    \n",
    "    weight = np.empty([n_samples, len_seq_half])\n",
    "    weight2 = np.empty([n_samples, len_seq_half])\n",
    "    weight.fill(std_1)\n",
    "    weight2.fill(std_2)\n",
    "    weight = np.append(weight, weight2, axis=1)\n",
    "\n",
    "    in_out_seq = [(torch.FloatTensor((seqs[index]).tolist()).to(device), torch.FloatTensor(weight[index].tolist()).to(device),\n",
    "               torch.FloatTensor(label.tolist()).to(device))\n",
    "              for index, label in enumerate(labels)]\n",
    "    return in_out_seq"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 579,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.37744383 0.10373804 0.04211682 ... 0.90386159 0.03179898 0.7379361 ]\n",
      " [0.57331014 0.97336213 0.97015168 ... 0.37972181 0.98205871 0.87147514]\n",
      " [0.70494839 0.09356418 0.66175278 ... 0.85751845 0.01811667 0.16906425]\n",
      " ...\n",
      " [0.86255429 0.58097579 0.98587284 ... 0.68581219 0.41190929 0.35070623]\n",
      " [0.67215342 0.31004733 0.23649079 ... 0.86134658 0.5815446  0.53114185]\n",
      " [0.04723546 0.48456673 0.60508312 ... 0.03955923 0.9493641  0.2903118 ]]\n"
     ]
    }
   ],
   "source": [
    "in_out_seq = create_sample(n_samples,length_sample,label_std,seq_std,label_offset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 569,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "unweighted 0.015925927\n",
      "weighted 0.010284853337999798\n"
     ]
    }
   ],
   "source": [
    "array = []\n",
    "array_weighted = []\n",
    "for seq, weight, label in in_out_seq:\n",
    "    average = torch.mean(seq)\n",
    "    weight = 1.0/(weight**2)\n",
    "    weighted_average = np.average(seq.tolist(), weights = weight)\n",
    "    \n",
    "    array.append((average-label[0].item())**2)\n",
    "    array_weighted.append((weighted_average-label[0].item())**2)\n",
    "    \n",
    "print(\"unweighted\",  np.mean(array))\n",
    "print(\"weighted\", np.mean(array_weighted))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 539,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LSTM(\n",
      "  (lstm): LSTM(1, 16)\n",
      "  (linear): Linear(in_features=16, out_features=1, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "model=one_to_one_lstm.LSTM(hidden_layer_size=16,output_size=1).to(device)\n",
    "#loss_function=nn.CrossEntropyLoss()\n",
    "#optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)\n",
    "loss_function=nn.MSELoss()\n",
    "optimizer=torch.optim.SGD(model.parameters(), lr=1e-2)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 553,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch:   0 loss: 0.01034392 lr: 0.01000000\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "optimizer=torch.optim.SGD(model.parameters(), lr=1e-3)\n",
    "for i in range(400):\n",
    "    in_out_seq = create_sample(n_samples,length_sample,label_std,seq_std,label_offset)\n",
    "    model.train(in_out_seq,optimizer,loss_function,device,epochs=1,lr=1e-2,draw_fig=True)\n",
    "    if i ==200: optimizer=torch.optim.SGD(model.parameters(), lr=1e-4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.004216659"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 570,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.010590937057464963\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.010590937"
      ]
     },
     "execution_count": 570,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loss, pred=model.test(in_out_seq,loss_function,device)\n",
    "loss_2=[]\n",
    "for index, label_seq in enumerate(in_out_seq):\n",
    "    #print(pred[index],label_seq[1])\n",
    "    loss_2.append((pred[index]-label_seq[2])**2)\n",
    "print(np.mean(loss))\n",
    "np.mean(loss_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 417,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 500\n",
    "length_sample = 8\n",
    "labels = np.random.rand(n_samples, 1)\n",
    "seqs = (np.random.rand(n_samples, length_sample)-.5)/2+labels\n",
    "in_out_seq = [(torch.FloatTensor((seqs[index]).tolist()).to(device),\n",
    "               torch.FloatTensor(label.tolist()).to(device))\n",
    "              for index, label in enumerate(labels)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
