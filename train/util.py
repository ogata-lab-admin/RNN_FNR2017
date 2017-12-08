#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np

def data_load(dir):
	fs = os.listdir(dir)
	fs.sort()
	print "loading", len(fs), "files..."
	dataset = []
	for fn in fs:
		data = np.loadtxt("%s/%s" %(dir,fn), dtype=np.float32)
		dataset.append(data)
	n_dataset = len(dataset)
	return dataset, n_dataset

def make_batch(dataset, batch_size):
    min_length_in_batch = 1000
    for b in range(batch_size):
        if dataset[b].shape[0] < min_length_in_batch:
            min_length_in_batch = dataset[b].shape[0]
    batch = np.full((batch_size, min_length_in_batch, dataset[0].shape[1]), -1.0, dtype=np.float32)
    for b in range(batch_size):
        batch[b,:,:] = dataset[b][:min_length_in_batch]
    batch = batch.transpose((1,0,2))
    return batch

def make_log_file(logtype = "error"):
    if logtype == "error":
        if os.path.exists("error.log"):
            log = np.loadtxt("error.log")
        else:
            log = np.zeros((0))
    return log
