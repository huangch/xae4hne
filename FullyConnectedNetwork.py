import tensorflow as tf

def FullyConnectedNetwork(
    dimensions = [256, 128, 64, 2],
    init_weight = None,
    init_bias = None,    
    ):
    
    y_ = tf.placeholder(tf.float32, [None, dimensions[-1]])
    x = tf.placeholder(tf.float32, [None, dimensions[0]])
    
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    
    input = tf.nn.dropout(x, keep_prob=keep_prob_input)
    
    weight_list = []
    bias_list = []
    
    for d in range(len(dimensions)-1):
        if init_weight != None:
            W = tf.Variable(tf.constant(init_weight[d]))
        else:
            W = tf.Variable(tf.truncated_normal([dimensions[d], dimensions[d+1]], stddev=0.01))
        
        if init_bias != None:
            b = tf.Variable(tf.constant(init_bias[d]))
        else:        
            b = tf.Variable(tf.constant(0.0, shape=[dimensions[d+1]]))
        
        weight_list.append(W)
        bias_list.append(b)
        
        if d == len(dimensions)-2:
            y = tf.nn.softmax(tf.matmul(input, W) + b)
            
        else:        
            # beta = tf.Variable(initial_value=1.0, trainable=True)
            # linear = tf.matmul(input, W) + b            
            # h = linear * tf.nn.sigmoid(beta*linear)
            
            h = tf.nn.relu(tf.matmul(input, W) + b)
            input = tf.nn.dropout(h, keep_prob) 

    # define the loss function
    cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
    prediction = tf.argmax(y, 1)
    # define training step and accuracy
    # train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    return {
        'cost': cost, 
        'prediction': prediction, 
        'accuracy': accuracy, 
        'x': x, 'y': y, 'y_': y_, 
        'keep_prob_input': keep_prob_input, 'keep_prob': keep_prob,
        'weight': weight_list,
        'bias': bias_list,
    }
    

