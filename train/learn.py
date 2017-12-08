#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
from chainer import cuda

sys.path.append(os.path.split(os.getcwd())[0])
from train_env import args, dataset, n_dataset, conf
from lstm import LSTM
from util import *

# batch
batch_size = n_dataset
io_size = dataset[0].shape[1]
batch = make_batch(dataset, n_dataset)
in_batch = batch.copy()[:,:,conf.in_index]
teach_batch = batch.copy()[:,:,conf.out_index]
del batch

# model
lstm = LSTM(conf, name="lstm")
lstm.load(args.model_dir)

# train
n_epoch = args.epoch
closed_time = args.closed_time
total_time = 0
error_log = make_log_file("error")
if conf.gpuid >= 0:
    cuda.check_cuda_available()
    cuda.get_device(conf.gpuid).use()

for epoch in xrange(n_epoch):
    start_time = time.time()
    sum_loss = lstm.train(in_batch, teach_batch, closed_time)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    sys.stdout.write("\r")
    print "epoch: {:d} loss: {:f} time: {:f} min total_time: {:f} ".format(epoch, float(sum_loss), elapsed_time, total_time)
    sys.stdout.flush()
    error_log = np.r_[error_log, np.array([sum_loss])]

lstm.save(args.model_dir, n_epoch)
np.savetxt("error.log", error_log)
