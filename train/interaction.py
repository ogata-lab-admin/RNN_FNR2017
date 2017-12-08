#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
from chainer import cuda

sys.path.append(os.path.split(os.getcwd())[0])
from test_env import args, dataset, n_dataset, conf
from lstm import LSTM
from util import *

# batch
batch_size = n_dataset
io_size = dataset[0].shape[1]
batch = make_batch(dataset, n_dataset)

# model
lstm = LSTM(conf, name="lstm")
lstm.load(args.model_dir)

if conf.gpuid >= 0:
    cuda.check_cuda_available()
    cuda.get_device(conf.gpuid).use()

# validate
lstm.reset_state()
c_init, h_init = lstm.get_init_state()
outs = batch.copy()
cs = np.zeros((outs.shape[0], outs.shape[1], conf.lstm_hidden_units[0]), dtype=np.float32)
hs = np.zeros((outs.shape[0], outs.shape[1], conf.lstm_hidden_units[0]), dtype=np.float32)
cs[0] = cuda.to_cpu(c_init[0].data)
hs[0] = cuda.to_cpu(h_init[0].data)
lstm.set_init_state()

for n in xrange(batch.shape[0]-1):
    c0 = batch[n, :, conf.in_index].T
    if n > 1:
        c0[:,conf.out_index] = cuda.to_cpu(c1.data)
    c1 = lstm.predict(c0, test=True)
    c, h = lstm.get_state()
    outs[n+1, :, conf.out_index] = cuda.to_cpu(c1.data).T
    cs[n+1] = cuda.to_cpu(c[0].data)
    hs[n+1] = cuda.to_cpu(h[0].data)

if not os.path.exists("result"):
    os.mkdir("result")

for i in range(batch.shape[1]):
    np.savetxt("result/result%.6d.txt" %i, outs[:,i,:])
