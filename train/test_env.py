# -*- coding: utf-8 -*-
import sys, os
import argparse

from util import *
from lstm import Conf

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--gpuid", type=int, default=-1)
parser.add_argument("--fc_output_type", type=int, default=2) #default: continuous
parser.add_argument("--lstm_apply_batchnorm", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="../target/test")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--closed_time", type=int, default=1)
args = parser.parse_args()

dataset, n_dataset = data_load(args.data_dir)

conf = Conf()
conf.gpuid = args.gpuid
conf.lstm_hidden_units = [100]
conf.fc_hidden_units = []
conf.fc_output_type = args.fc_output_type
conf.inst_index = [0,1,2,3,4,5,6,7,8]
conf.vision_index = [9,10,11]
conf.joint_index = [12,13]
conf.in_index = conf.inst_index + conf.vision_index + conf.joint_index
conf.out_index = conf.joint_index
conf.in_size = len(conf.in_index)
conf.out_size = len(conf.out_index)
conf.n_seq = n_dataset
conf.learning_rate = args.learning_rate
