# -*- coding: utf-8 -*-
import os, time, glob
import numpy as np
import chainer
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from chainer import Link

activations = {
    "sigmoid": F.sigmoid, 
    "tanh": F.tanh, 
    "softplus": F.softplus, 
    "relu": F.relu, 
    "leaky_relu": F.leaky_relu, 
    "elu": F.elu
}

class Conf:
    def __init__(self):
        self.gpuid = -1

        self.inst_index = range(0,1)
        self.vision_index = range(1,2)
        self.joint_index =  range(2,3)
        self.in_index = range(0,3)
        self.out_index = range(2,3)
        self.in_size = 3
        self.out_size = 1
        self.n_seq = 1
        self.lstm_hidden_units = [1000]
        self.fc_hidden_units = []
        self.fc_activation_function = "tanh"
        self.fc_output_type = LSTM.OUTPUT_TYPE_CONTINUOUS
        self.learning_rate = 0.001
        self.gradient_momentum = 0.95

    def check(self):
        if len(self.lstm_hidden_units) < 1:
            raise Exception("You need to add one or more hidden layers to LSTM network.")

class LSTMNetwork(chainer.Chain):
    def __init__(self, **layers): 
        super(LSTMNetwork, self).__init__(**layers)
        self.n_layers = 0
        self.n_seq = 0
        self.activation_function = "tanh"

    def forward_one_step(self, x, test):
        f = activations[self.activation_function]
        chain = [x]

        for i in range(self.n_layers):
            net = getattr(self, "layer_%i" % i)
            u = net(chain[-1])
            output = u
            chain.append(output)
            
        return chain[-1]

    def set_init_state(self):
        for i in range(self.n_layers):
            getattr(self, "layer_%i" % i).c = (getattr(self, "layer_%i_c_init" %i).state)
            getattr(self, "layer_%i" % i).h = (getattr(self, "layer_%i_h_init" %i).state)

    def arbitrarily_set_init_state(self, size):
        for i in range(self.n_layers):
            getattr(self, "layer_%i" % i).c = Variable(self.xp.asarray(np.zeros((size, getattr(self, "layer_%i" % i).upward.W.data.shape[0]/4)), dtype=np.float32))
            getattr(self, "layer_%i" % i).h = Variable(self.xp.asarray(np.zeros((size, getattr(self, "layer_%i" % i).upward.W.data.shape[0]/4)), dtype=np.float32))
		

    def copy_init_state(self):
        for i in range(self.n_layers):
            getattr(self, "layer_%i" % i).c.data[:] = (getattr(self, "layer_%i_c_init" %i).state.data[0])
            getattr(self, "layer_%i" % i).h.data[:] = (getattr(self, "layer_%i_h_init" %i).state.data[0])

    def reset_state(self):
        for i in range(self.n_layers):
            getattr(self, "layer_%i" % i).reset_state()

    def add_noise2cell(self, size, amount):
        for i in range(self.n_layers):
            getattr(self, "layer_%i" % i).c += Variable(self.xp.asarray(np.random.randn(size, getattr(self, "layer_%i" % i).upward.W.data.shape[0]/4)*amount, dtype=np.float32))

    def get_state(self):
        c_states, h_states = [],[]
        for i in range(self.n_layers):
            c_states.append(getattr(self, "layer_%i" % i).c)
            h_states.append(getattr(self, "layer_%i" % i).h)
        return c_states,h_states

    def get_init_state(self):
        c_init, h_init = [], []
        for i in range(self.n_layers):
            c_init.append(getattr(self, "layer_%i_c_init" % i).state)
            h_init.append(getattr(self, "layer_%i_h_init" % i).state)
        return c_init,h_init

    def __call__(self, x, test=False):
        return self.forward_one_step(x, test=test)

class FullyConnectedNetwork(chainer.Chain):
    def __init__(self, **layers):
        super(FullyConnectedNetwork, self).__init__(**layers)
        self.n_layers = 0
        self.activation_function = "tanh"

    def forward_one_step(self, x, test):
        f = activations[self.activation_function]
        chain = [x]

        for i in range(self.n_layers):
            u = chain[-1]
            u = getattr(self, "layer_%i" % i)(u)
            output = f(u)
            chain.append(output)

        return chain[-1]

    def __call__(self, x, test=False):
        return self.forward_one_step(x, test=test)

class LSTM:
    OUTPUT_TYPE_SOFTMAX = 1
    OUTPUT_TYPE_CONTINUOUS = 2
    def __init__(self, conf, name="lstm"):
        self.output_type = conf.fc_output_type
        self.lstm, self.fc = self.build(conf)
        self.name = name
        self.pre_epoch = 0
        self.optimizer_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
        self.optimizer_lstm.setup(self.lstm)
        self.optimizer_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

        self.optimizer_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
        self.optimizer_fc.setup(self.fc)
        self.optimizer_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

        self.in_index = conf.in_index
        self.out_index = conf.out_index

    def build(self, conf):
        conf.check()
        wscale = 1.0

        lstm_attributes = {}
        lstm_units = [(conf.in_size, conf.lstm_hidden_units[0])]
        lstm_units += zip(conf.lstm_hidden_units[:-1], conf.lstm_hidden_units[1:])
        
        for i, (n_in, n_out) in enumerate(lstm_units):
            lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)
            lstm_attributes["layer_%i_c_init" %i] = Link(state=(conf.n_seq, n_out))
            lstm_attributes["layer_%i_h_init" %i] = Link(state=(conf.n_seq, n_out))
            lstm_attributes["layer_%i_c_init" %i].state.data.fill(0)
            lstm_attributes["layer_%i_h_init" %i].state.data.fill(0)     
		
        lstm = LSTMNetwork(**lstm_attributes) 
        lstm.n_layers = len(lstm_units) 
        lstm.n_seq = conf.n_seq
            
        if conf.gpuid >= 0:
            lstm.to_gpu(device=conf.gpuid)

        fc_attributes = {}
        if len(conf.fc_hidden_units) == 0:
            fc_units = [(conf.lstm_hidden_units[-1], conf.out_size)]						
        else:
            fc_units = [(conf.lstm_hidden_units[-1], conf.fc_hidden_units[0])]
            fc_units += zip(conf.fc_hidden_units[:-1], conf.fc_hidden_units[1:])
            fc_units += [(conf.fc_hidden_units[-1], conf.out_size)]

        for i, (n_in, n_out) in enumerate(fc_units):
            fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)

        fc = FullyConnectedNetwork(**fc_attributes) #full connectionパートのinstance
        fc.n_layers = len(fc_units)
        fc.activation_function = conf.fc_activation_function
        if conf.gpuid >= 0:
            fc.to_gpu(device=conf.gpuid)

       	return lstm, fc

    def __call__(self, x, test=False, softmax=True):
        output = self.lstm(x, test=test)
        output = self.fc(output, test=test)
        if softmax and self.output_type == self.OUTPUT_TYPE_SOFTMAX:
            output = F.softmax(output)
        return output

    @property
    def xp(self):
        return np if self.lstm.layer_0._cpu else cuda.cupy

    @property
    def gpu(self):
        if hasattr(cuda, "cupy"):
            return True if self.xp is cuda.cupy else False
        return False

    def reset_state(self):
        self.lstm.reset_state()

    def set_init_state(self):
        self.lstm.set_init_state()

    def arbitrarily_set_init_state(self, size=1):
        self.lstm.arbitrarily_set_init_state(size)

    def copy_init_state(self):
        self.lstm.copy_init_state()

    def add_noise2cell(self, size, amount):
        self.lstm.add_noise2cell(size, amount)

    def get_state(self):
        c, h = self.lstm.get_state()
        return c,h

    def get_init_state(self):
        return self.lstm.get_init_state()

    def predict(self, one, test=True, argmax=False):
        xp = self.xp
        c0 = Variable(xp.asarray(one, dtype=np.float32))
        if self.output_type == self.OUTPUT_TYPE_SOFTMAX:
            output = self(c0, test=test, softmax=True)
            if hasattr(cuda, "cupy") and xp is cuda.cupy:
                output.to_cpu()
            if argmax:
                ids = np.argmax(output.data, axis=1)
            else:
                ids = [np.random.choice(np.arange(output.data.shape[1]), p=output.data[0])]
        elif self.output_type == self.OUTPUT_TYPE_CONTINUOUS:
            output = self(c0, test=test, softmax=False)
        return output

    def train(self, in_batch, teach_batch, closed_time, test=False):
        if closed_time >= 0:
            return self.closed_train(in_batch, teach_batch, closed_time)

        self.reset_state()
        self.set_init_state()
        xp = self.xp
        sum_loss = 0
        zeros = Variable(xp.asarray(np.zeros((256, 1), dtype=np.float32)))

        for c0, c1 in zip(in_batch[:-1], teach_batch[1:]):
            c0 = Variable(xp.asarray(c0, dtype=np.float32))
            c1 = Variable(xp.asarray(c1, dtype=np.float32))
            output = self(c0, test=test, softmax=False)
            if self.output_type == self.OUTPUT_TYPE_SOFTMAX:
                loss = F.softmax_cross_entropy(output, c1)
            elif self.output_type == self.OUTPUT_TYPE_CONTINUOUS:
                loss = F.mean_squared_error(output, c1)
            else:
                raise Exception()
            sum_loss += loss
        self.zero_grads()
        sum_loss.backward()
        self.update()
        if self.gpu:
            sum_loss.to_cpu()
        return sum_loss.data

    def closed_train(self, in_batch, teach_batch, closed_time, test=False):
        self.reset_state()
        self.set_init_state()
        xp = self.xp
        sum_loss = 0
        stamp = 0
        pre_out = None

        for c0, c1 in zip(in_batch[:-1], teach_batch[1:]):
            if stamp > closed_time:
                c0[:, self.out_index] = cuda.to_cpu(output.data)
            c0 = Variable(xp.asarray(c0, dtype=np.float32))
            c1 = Variable(xp.asarray(c1, dtype=np.float32))
            output = self(c0, test=test, softmax=False)
            if self.output_type == self.OUTPUT_TYPE_SOFTMAX:
                loss = F.softmax_cross_entropy(output, c1)
            elif self.output_type == self.OUTPUT_TYPE_CONTINUOUS:
                loss = F.mean_squared_error(output, c1)
            else:
                raise Exception()
            sum_loss += loss
            stamp += 1
        self.zero_grads()
        sum_loss.backward()
        self.update()
        if self.gpu:
            sum_loss.to_cpu()
        return sum_loss.data

    def zero_grads(self):
        self.optimizer_lstm.zero_grads()
        self.optimizer_fc.zero_grads()

    def update(self):
        self.optimizer_lstm.update()
        self.optimizer_fc.update()

    def should_save(self, prop):
        if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
            return True
        return False

    def load(self, dir=None):
        if dir is None:
            raise Exception()
        for attr in vars(self):
            prop = getattr(self, attr)
            if self.should_save(prop):
                files = glob.glob(dir + "/%s_%s_*" % (self.name, attr))
                files.sort()
                if len(files) > 0:
                    filename = files[-1]
                    self.pre_epoch = int(filename.split("_")[-1].split(".")[0]) 
                    print "loading",  filename
                    serializers.load_hdf5(filename, prop)
                else:
                    print "no file to load exists."
        print "model loaded."

    def save(self, dir=None, epoch = 0):
        if dir is None:
            raise Exception()
        try:
            os.mkdir(dir)
        except:
            pass
        for attr in vars(self):
            prop = getattr(self, attr)
            if self.should_save(prop):
                serializers.save_hdf5(dir + "/%s_%s_%.6d.hdf5" % (self.name, attr, self.pre_epoch+epoch), prop)
        print "model saved."
