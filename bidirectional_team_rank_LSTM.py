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
   "execution_count": 660,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<module 'one_to_one_lstm' from '/Users/ryandmueller/Documents/cfb_pytorch/CFB_pytorch_toy_minimizer/one_to_one_lstm.py'>"
      ]
     },
     "execution_count": 660,
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
   "execution_count": 701,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 500\n",
    "length_sample = 5\n",
    "label_std = 14\n",
    "seq_std = 14\n",
    "label_offset = 0\n",
    "label_std = label_std/40\n",
    "seq_std = seq_std/40\n",
    "label_offset =.5\n",
    "def create_sample(n_samples,length_sample,label_std,seq_std, label_offset):\n",
    "    labels = np.random.normal(label_offset, label_std, size=(n_samples, 1))\n",
    "    \n",
    "    std_array = np.abs(np.random.normal(seq_std, seq_std/2, size=(n_samples, length_sample)))\n",
    "    #std_array.fill(.1)\n",
    "    seqs = np.random.normal(0, std_array, size=(n_samples, length_sample))\n",
    "    seqs = (seqs+labels) \n",
    "\n",
    "    in_out_seq = [(torch.FloatTensor((seqs[index]).tolist()).to(device), torch.FloatTensor(std_array[index].tolist()).to(device),\n",
    "               torch.FloatTensor(label.tolist()).to(device))\n",
    "              for index, label in enumerate(labels)]\n",
    "    in_out_seq_comb = []\n",
    "    for seq, weight, label in in_out_seq:\n",
    "        in_out_seq_comb.append((torch.stack([seq,weight],dim=1),label))\n",
    "    \n",
    "    return in_out_seq,in_out_seq_comb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 719,
   "metadata": {},
   "outputs": [],
   "source": [
    "in_out_seq,in_out_seq_comb = create_sample(n_samples,length_sample,label_std,seq_std,label_offset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 720,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "unweighted 0.028173199\n",
      "weighted 0.012290785545207659\n"
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
   "execution_count": 709,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LSTM(\n",
      "  (lstm): LSTM(2, 16, num_layers=2)\n",
      "  (linear): Linear(in_features=16, out_features=1, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "model=one_to_one_lstm.LSTM(input_size=2,hidden_layer_size=16,output_size=1,num_layers=2).to(device)\n",
    "#loss_function=nn.CrossEntropyLoss()\n",
    "#optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)\n",
    "loss_function=nn.MSELoss()\n",
    "optimizer=torch.optim.Adam(model.parameters(), lr=1e-2)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 716,
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
      "epoch:   0 loss: 0.01715918 lr: 0.01000000\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-716-9bd731a1a658>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m400\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0min_out_seq\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0min_out_seq_comb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcreate_sample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mn_samples\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlength_sample\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlabel_std\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mseq_std\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlabel_offset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m     \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0min_out_seq_comb\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0moptimizer\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mloss_function\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlr\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1e-2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mdraw_fig\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mi\u001b[0m \u001b[0;34m==\u001b[0m\u001b[0;36m200\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0moptimizer\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptim\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mAdam\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1e-3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/Documents/cfb_pytorch/CFB_pytorch_toy_minimizer/one_to_one_lstm.py\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(self, train_inout_seq, optimizer, loss_function, device, epochs, lr, draw_fig)\u001b[0m\n\u001b[1;32m     49\u001b[0m         \u001b[0my_pred\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mview\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     50\u001b[0m         \u001b[0msingle_loss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mloss_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 51\u001b[0;31m         \u001b[0msingle_loss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     52\u001b[0m         \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msingle_loss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     53\u001b[0m         \u001b[0moptimizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.7/site-packages/torch/tensor.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(self, gradient, retain_graph, create_graph)\u001b[0m\n\u001b[1;32m    196\u001b[0m                 \u001b[0mproducts\u001b[0m\u001b[0;34m.\u001b[0m \u001b[0mDefaults\u001b[0m \u001b[0mto\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    197\u001b[0m         \"\"\"\n\u001b[0;32m--> 198\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mautograd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgradient\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    199\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    200\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mregister_hook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.7/site-packages/torch/autograd/__init__.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables)\u001b[0m\n\u001b[1;32m     98\u001b[0m     Variable._execution_engine.run_backward(\n\u001b[1;32m     99\u001b[0m         \u001b[0mtensors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgrad_tensors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 100\u001b[0;31m         allow_unreachable=True)  # allow_unreachable flag\n\u001b[0m\u001b[1;32m    101\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    102\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
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
    "optimizer=torch.optim.Adam(model.parameters(), lr=1e-2)\n",
    "for i in range(400):\n",
    "    in_out_seq,in_out_seq_comb = create_sample(n_samples,length_sample,label_std,seq_std,label_offset)\n",
    "    model.train(in_out_seq_comb,optimizer,loss_function,device,epochs=1,lr=1e-2,draw_fig=True)\n",
    "    if i ==200: optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 700,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LSTM(\n",
      "  (lstm): LSTM(2, 16, num_layers=2)\n",
      "  (linear): Linear(in_features=16, out_features=1, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 718,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.011991184183387843\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.011991184"
      ]
     },
     "execution_count": 718,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loss, pred=model.test(in_out_seq_comb,loss_function,device)\n",
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
