# RNN_FNR2017

Recurrent Neural Network implemented by Chainer

Last updated: Dec. 8th, 2017.

Copyright (c) 2017, Tatsuro Yamada <<yamadat@idr.ias.sci.waseda.ac.jp>>

## Requirements
- Python 2.7 (NO supports for 3.4 nor 3.5)
- Chainer 1.24
- NumPy 1.11

## Implementation
Recurrent Neural Network with LSTM cells

## Example
**Flag game**
- Data download
    * You can download the data from http://ogata-lab.jp/projects/cognitive-robotics-group.html
- Data format
    * The flag game is represented as a sequence of 14 dim. vectors.
      * columns 0-8 represent a word (1-hot encoding).
      * columns 9-11 represent visual information (RGB).
      * columns 12,13 represent the robot's left and right arm, respectively.
- Preparation
    * Putting training sequences in target/train/
    * Putting test sequences in target/test/
- Training
    * $ cd train
    * $ python learn.py --data_dir ../target/train --gpuid #
    * (If you don't use GPU, please delete the gpuid option.)
    * After training, the model will be saved in model/
- Test (Generating joint sequences that respond to instructions)
    * $ python interaction.py --data_dir ../target/test --gpuid #
    * Generated sequences will be put in result/
- Evaluation of generated data
    * $ python validate.py result ../target/test
    * The evaluation results are saved as result.dump
- Quantifiation of the result
    * $ python correct_rate.py
    * The correct rate for each type of instructions will be displayed