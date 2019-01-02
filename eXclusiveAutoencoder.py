'''eXclusive Autoencoder

Huang et al., Dec 2018
'''
# %% Imports
import tensorflow as tf
import numpy as np
import math
        
# %% Autoencoder definition
def eXclusiveAutoencoder(
        input_dimensions = 784,
        layers = [
            {
                'n_channels': 144,
                'reconstructive_regularizer': 1.0, 
                'weight_decay': 1.0, 
                'sparse_regularizer': 1.0, 
                'sparsity_level': 0.05,
                'exclusive_regularizer': 10.0,
                'exclusive_scale': 1.0,
                'corrupt_prob': 1.0,
                'tied_weight': True,
                'exclusive_type': 'logcosh',
                'exclusive_scale': 1.0,    
                'gaussian_mean': 0.0,    
                'gaussian_std': 0.0,                
                'encode':'sigmoid', 'decode':'linear',
                'pathways': [
                    range(0, 72),
                    range(0, 144),
                ],
            },                                                                                                 
        ],
        global_corrupt_prob = 1.0,
        global_reconstructive_regularizer = 1.0,
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,
        ):
    '''Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    '''
    
    dimensions = [input_dimensions]
    for layer in layers:
        dimensions.append(layer['n_channels'])
        assert len(layer['pathways']) == len(layers[0]['pathways']), 'Ambiguous pathway definitions over layers.'
        
    # %% input to the network
    training_x = []
    for pathway_i in range(len(layers[0]['pathways'])):
        training_x.append(tf.placeholder(tf.float32, [None, dimensions[0]], name='training_x' + str(pathway_i)))
        
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
     
    training_current_input = []
    training_current_corrupt_input = []
    
    for pathway_i in range(len(layers[0]['pathways'])):
        training_current_input.append(
            training_x[pathway_i]
        )
        
        training_current_corrupt_input.append(
            corrupt(training_x[pathway_i]) * global_corrupt_prob \
                + training_x[pathway_i] * (1.0 - global_corrupt_prob) \
                if global_corrupt_prob != None else training_x[pathway_i]
        )
    
    current_input = x

    # %% Build the encoder
    encoder_weight = [] 
    encoder_bias = []            
    training_encoder_output_list = []
    training_encoder_input_list = []
    layerwise_z = []
    
    for layer_i, (n_input, n_output) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        if init_encoder_weight != None:
            W = tf.Variable(tf.constant(init_encoder_weight[layer_i]))            
        else:
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                - 1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        
        if init_encoder_bias != None:
            b = tf.Variable(tf.constant(init_encoder_bias[layer_i]))
        else:
            b = tf.Variable(tf.zeros([n_output]))
        
        encoder_weight.append(W)
        encoder_bias.append(b)
        
        training_encoder_input_list.append([]) 
        training_encoder_corrupt_output = []
        training_encoder_output = []
        
        for pathway_i in range(len(layers[layer_i]['pathways'])):           
            # training_encoder_input_list[-1].append(None)   
                        
            # training_encoder_input_list[-1][-1] \
            #     = corrupt(training_current_input[pathway_i]) * layers[layer_i]['corrupt_prob'] \
            #     + training_current_input[pathway_i] * (1 - layers[layer_i]['corrupt_prob']) \
            #     if layers[layer_i]['corrupt_prob'] != None else training_current_input[pathway_i]
                
            # training_encoder_input_list[-1][-1] = training_current_input[pathway_i]
                                     
            # training_encoder_input = corrupt(training_current_input[pathway_i]) * layers[layer_i]['corrupt_prob'] \
            #     + training_current_input[pathway_i] * (1 - layers[layer_i]['corrupt_prob']) \
            #     if layers[layer_i]['corrupt_prob'] != None else training_current_input[pathway_i]
                 
            # training_encoder_corrupt_input_list[-1].append(training_current_corrupt_input[pathway_i])
            training_encoder_input_list[-1].append(training_current_input[pathway_i])
            
            # a = activate_function(
            #     tf.matmul(training_current_input[pathway_i], W) + b,
            #     layers[layer_i]['encode'],
            # )
            
            corrupt_a = activate_function(
                tf.matmul(training_current_corrupt_input[pathway_i], W) + b,
                layers[layer_i]['encode'],
            )
            
            training_encoder_corrupt_output.append(corrupt_a)
            
            a = activate_function(
                tf.matmul(
                    # training_current_input[pathway_i], 
                    corrupt(training_current_input[pathway_i]) * layers[layer_i]['corrupt_prob'] \
                        + training_current_input[pathway_i] * (1.0 - layers[layer_i]['corrupt_prob']) \
                        if layers[layer_i]['corrupt_prob'] != None else training_current_input[pathway_i],
                 
                    W
                ) + b,
                layers[layer_i]['encode'],
            )
            
            training_encoder_output.append(a)

        # training_encoder_corrupt_output_list.append(training_encoder_corrupt_output)        
        training_current_corrupt_input = training_encoder_corrupt_output
        
        training_encoder_output_list.append(training_encoder_output)
        
        output = activate_function( tf.matmul(current_input, W) + b, layers[layer_i]['encode'])
        layerwise_z.append(output)
        current_input = output

    # %% latent representation
    training_z = training_encoder_corrupt_output
    z = current_input
    
    decoder_weight = []
    decoder_bias = []
    training_decoder_corrupt_output_list = []
    layerwise_training_decoder_output_list = []
    # %% Build the decoder using the same weights
    for layer_i, (n_input, n_output) in enumerate(zip(dimensions[::-1][:-1], dimensions[::-1][1:])):
        
        if init_decoder_weight != None:
            W = tf.Variable(tf.constant(init_decoder_weight[::-1][layer_i]))            
        else:
            if layers[layer_i]['tied_weight']:
                W = tf.transpose(encoder_weight[::-1][layer_i])
            else:
                W = tf.Variable(tf.random_uniform([n_input, n_output],
                    - 1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
        
        if init_decoder_bias != None:
            b = tf.Variable(tf.constant(init_decoder_bias[::-1][layer_i]))
        else:
            b = tf.Variable(tf.zeros([n_output]))
            
        decoder_weight.append(W)  
        decoder_bias.append(b)            
        
        training_decoder_corrupt_output = []
        layerwise_training_decoder_output = []
        
        for pathway_i in range(len(layers[::-1][layer_i]['pathways'])):
            corrupt_a = activate_function( 
                tf.matmul(
                    tf.gather(training_current_corrupt_input[pathway_i], layers[::-1][layer_i]['pathways'][pathway_i], axis=1),
                    tf.gather(W, layers[::-1][layer_i]['pathways'][pathway_i])) + b,
                layers[::-1][layer_i]['decode'],
            ) 
#             a = activate_function( 
#                 tf.matmul(
#                     training_current_input[pathway_i],
#                     W) + b,
#                 layers[::-1][layer_i]['decode'],
#             )             
            layerwise_a = activate_function( 
                tf.matmul(
                    tf.gather(training_encoder_output_list[::-1][layer_i][pathway_i], layers[::-1][layer_i]['pathways'][pathway_i], axis=1),
                    tf.gather(W, layers[::-1][layer_i]['pathways'][pathway_i])) + b,
                layers[::-1][layer_i]['decode'],
            )
#             layerwise_a = activate_function( 
#                 tf.matmul(
#                     training_encoder_output_list[::-1][layer_i][pathway_i],
#                     W
#                     ) + b,
#                 layers[::-1][layer_i]['decode'],
#             ) 
            training_decoder_corrupt_output.append(corrupt_a)
            layerwise_training_decoder_output.append(layerwise_a)
            
        training_decoder_corrupt_output_list.append(training_decoder_corrupt_output)
        layerwise_training_decoder_output_list.append(layerwise_training_decoder_output)
        training_current_corrupt_input = training_decoder_corrupt_output
        
        output = activate_function(
            tf.matmul(current_input, W) + b,
            layers[::-1][layer_i]['decode']
        )
        
        current_input = output

    # %% now have the reconstruction through the network
    training_y = training_current_corrupt_input    
    y = current_input

    layerwise_training_decoder_output_list.reverse()      
    training_decoder_corrupt_output_list.reverse()

    layerwise_y = []
    for layer_i in range(len(dimensions)-1):        
        layerwise_current_input = layerwise_z[::-1][layer_i]
        
        for layer_j in range(layer_i, len(dimensions)-1):
            W = decoder_weight[layer_j]
            b = decoder_bias[layer_j]
            layerwise_output = activate_function( tf.matmul(layerwise_current_input, W) + b, layers[::-1][layer_i]['decode'],)
            layerwise_current_input = layerwise_output
            
        layerwise_y.append(layerwise_current_input)
    
    decoder_weight.reverse()
    decoder_bias.reverse()
    layerwise_y.reverse()
   
    cost = {}
        
    cost['reconstruction_error'] = tf.constant(0.0)
    for pathway_i in range(len(layers[0]['pathways'])):
        if training_x[pathway_i] != None and training_y[pathway_i] != None:
            cost['reconstruction_error'] = tf.add(
                cost['reconstruction_error'], 
                global_reconstructive_regularizer * 0.5 * tf.reduce_mean(
                    tf.square(
                        tf.subtract(
                            training_x[pathway_i],
                            training_y[pathway_i]
                        )
                    )
                )
            )

    cost['weight_decay'] = tf.constant(0.0)   
    for layer_i in range(len(dimensions)-1):
        cost_encoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(encoder_weight[layer_i]))
        
        if layers[layer_i]['tied_weight']:
            cost['weight_decay'] = tf.add(cost['weight_decay'], cost_encoder_weight_decay)
        else:
            cost_decoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(decoder_weight[layer_i]))
            cost['weight_decay'] = tf.add(cost['weight_decay'], cost_encoder_weight_decay+cost_decoder_weight_decay)
            
    cost['exclusivity'] = tf.constant(0.0)
    for layer_i in range(len(dimensions)-1):
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):
            exclusivity = np.setdiff1d(range(layers[layer_i]['n_channels']), layers[layer_i]['pathways'][pathway_i]).tolist()
            if exclusivity != [] and encoder_pathway_output != None:
                if layers[layer_i]['exclusive_type'] == 'pow4':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity), 4),
                                )
                            )
                        )                        
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity), 4),
                                        tf.pow(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']), 4)
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'exp':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']))),
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'logcosh':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )
                                )
                            )
                        )
    
    cost['sparsity'] = tf.constant(0.0)
    for layer_i in range(len(dimensions)-1):
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):                
            if layers[layer_i]['pathways'][pathway_i] != None and encoder_pathway_output != None:
                cost['sparsity'] = tf.add(cost['sparsity'], layers[layer_i]['sparse_regularizer'] * kl_divergence(layers[layer_i]['sparsity_level'], tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), layers[layer_i]['pathways'][pathway_i])))
    
    cost['total'] = cost['reconstruction_error'] + cost['weight_decay'] + cost['exclusivity'] + cost['sparsity']

    layerwise_cost = []
    for layer_i in range(len(dimensions)-1):
        layerwise_cost.append({})
        
        layerwise_cost[layer_i]['reconstruction_error'] = tf.constant(0.0)
        for pathway_i in range(len(layers[0]['pathways'])):
            if training_encoder_input_list[layer_i][pathway_i] != None and layerwise_training_decoder_output_list[layer_i][pathway_i] != None:
                layerwise_cost[layer_i]['reconstruction_error'] = tf.add(
                    layerwise_cost[layer_i]['reconstruction_error'], 
                    layers[layer_i]['reconstructive_regularizer'] * 0.5 * tf.reduce_mean(
                        tf.square(
                            tf.subtract(
                                training_encoder_input_list[layer_i][pathway_i], 
                                layerwise_training_decoder_output_list[layer_i][pathway_i],
                            )
                        )
                    )
                )
                         
        layerwise_cost[layer_i]['weight_decay'] = tf.constant(0.0)
        layerwise_cost_encoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(encoder_weight[layer_i]))        
        if layers[layer_i]['tied_weight']:         
            layerwise_cost[layer_i]['weight_decay'] = tf.add(layerwise_cost[layer_i]['weight_decay'], layerwise_cost_encoder_weight_decay)
        else:                 
            layerwise_cost_decoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(decoder_weight[layer_i]))
            layerwise_cost[layer_i]['weight_decay'] = tf.add(layerwise_cost[layer_i]['weight_decay'], layerwise_cost_encoder_weight_decay+layerwise_cost_decoder_weight_decay)
                
        layerwise_cost[layer_i]['exclusivity'] = tf.constant(0.0) 
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):
            exclusivity = np.setdiff1d(range(layers[layer_i]['n_channels']), layers[layer_i]['pathways'][pathway_i]).tolist()               
            if exclusivity != [] and encoder_pathway_output != None:
                if layers[layer_i]['exclusive_type'] == 'pow4':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity), 4),
                                )
                            )
                        )
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity), 4),
                                        tf.pow(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']), 4)
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'exp':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'logcosh':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),                                                                         
                                )
                            )
                        )
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), exclusivity))),
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )                                    
                                )
                            )
                        )

        layerwise_cost[layer_i]['sparsity'] = tf.constant(0.0)
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):                
            if layers[layer_i]['pathways'][pathway_i] != None and encoder_pathway_output != None:
                layerwise_cost[layer_i]['sparsity'] = tf.add(layerwise_cost[layer_i]['sparsity'], layers[layer_i]['sparse_regularizer'] * kl_divergence(layers[layer_i]['sparsity_level'], tf.gather(tf.reduce_mean(encoder_pathway_output, [0]), layers[layer_i]['pathways'][pathway_i])))
                  
        layerwise_cost[layer_i]['total'] = layerwise_cost[layer_i]['reconstruction_error'] + layerwise_cost[layer_i]['weight_decay'] + layerwise_cost[layer_i]['exclusivity'] + layerwise_cost[layer_i]['sparsity']
         
    return {'training_x': training_x, 'training_z': training_z, 'training_y': training_y,
            'x': x, 'y': y, 'z': z, 
            'layerwise_y': layerwise_y, 'layerwise_z': layerwise_z, 
            'cost': cost, 'layerwise_cost': layerwise_cost, 
            'encoder_weight': encoder_weight, 'decoder_weight': decoder_weight,
            'encoder_bias': encoder_bias, 'decoder_bias': decoder_bias,
        }

# %%
def corrupt( x):
    '''Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    '''
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


def activate_function( linear, name, leak=0.2):
    if name == 'sigmoid':
        return tf.nn.sigmoid(linear, name='encoded')
    elif name == 'softmax':
        return tf.nn.softmax(linear, name='encoded')
    elif name == 'linear':
        return linear
    elif name == 'tanh':
        return tf.nn.tanh(linear, name='encoded')
    elif name == 'relu':
        return tf.nn.relu(linear, name='encoded')
    elif name == 'lrelu':
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * linear + f2 * abs(linear)
    elif name == 'swish':
        return linear * tf.nn.sigmoid(linear, name='encoded')
    elif name == 'bswish':
        beta = tf.Variable(initial_value=1.0, trainable=True, name='encoded')
        return linear * tf.nn.sigmoid(beta*linear)
        
         
def kl_divergence( p, p_hat):
    return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-10, 1.0)) - p * tf.log(tf.clip_by_value(p_hat, 1e-10, 1.0)) + (1 - p) * tf.log(tf.clip_by_value(1 - p, 1e-10, 1.0)) - (1 - p) * tf.log(tf.clip_by_value(1 - p_hat, 1e-10, 1.0)))

