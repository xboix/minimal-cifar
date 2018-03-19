import tensorflow as tf

from nets.MLP import MLP1 as net_MLP1
from nets.MLP import MLP3 as net_MLP3
from util import summary as summ


def MLP3(x, dropout_rate, opt, labels_id):
    return net_MLP3(x, opt, labels_id, dropout_rate)


def MLP1(x, dropout_rate, opt, labels_id):
    return net_MLP1(x, opt, labels_id, dropout_rate)

