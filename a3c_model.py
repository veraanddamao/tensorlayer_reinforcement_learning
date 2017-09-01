import tensorflow as tf
import tensorlayer as tl

def build_policy_networks(num_actions, agent_history_length, resized_width, resized_height,reuse=False):
    
    with tf.variable_scope("polic", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        s = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
        network = tl.layers.InputLayer(s,name='input')
        network = tl.layers.Conv2d(network,n_filter=16, filter_size=(8, 8), strides=(4, 4), act=tf.nn.relu, name='cnn1')
        network = tl.layers.Conv2d(network,n_filter=32, filter_size=(4, 4), strides=(2,2), act=tf.nn.relu, name='cnn2')
        network = tl.layers.FlattenLayer(network,name='flatten')
        share = tl.layers.DenseLayer(network,n_units=256, act=tf.nn.relu,name='Dense1')

        action_probs = tl.layers.DenseLayer(share, n_units=num_actions, act=tf.nn.softmax,name='action_probs')
        
        state_value = tl.layers.DenseLayer(share, n_units=1, act=tf.identity,name='state_value')
        p_out = action_probs.outputs
        v_out = state_value.outputs

    return s,p_out,v_out
