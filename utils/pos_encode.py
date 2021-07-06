import numpy as np 
import tensorflow as tf

def get_formula(pos, i, d_model):
    formula = 1/np.pow(10000, (2*(i//2)) / d_model)
    return pos * formula

def pos_encode(position, d_model):
    pos_encode_vec = get_formula (
        np.arange(position).reshape(-1,1),
        np.arange(d_model).reshape(1,-1),
        d_model
    )
    pos_encode_vec[:,0::2] = np.sin(pos_encode_vec[:,0::2])
    pos_encode_vec[:,1::2] = np.cos(pos_encode_vec[:,1::2])


    pos_encoding = tf.reshape( pos_encode_vec, ( position, d//2, 2))
    pos_encoding = tf.transpose( pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape( pos_encoding, (d_model, position))

    return pos_encoding