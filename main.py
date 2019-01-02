"""Tutorial on how to create an autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
# %% Imports
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import scipy.io
from display_network import display_color_network
from display_network import display_network
import glob
from PIL import Image
from random import randint
import random
from tensorflow.examples.tutorials.mnist import input_data
from eXclusiveAutoencoder import eXclusiveAutoencoder
# from SoftmaxClassifier import SoftmaxClassifier
from FullyConnectedNetwork import FullyConnectedNetwork
from scipy import io as spio
import cPickle as pickle
import os
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, classification_report, confusion_matrix, roc_curve, auc
import pyCudaSampling as pcs
import math
import csv
import matplotlib.pyplot as plt
import sys
import AMSGrad
import cv2

# RANDOM_SEED_MAGIC = 1234
RANDOM_SEED_MAGIC = random.randint(0,65536) 
PATCH_SIZE = 11
DATA_AUGMENTATION = 1
LABEL_LIST = ['Others', 'Lymphocyte']

# class LymphocyteDataset():
#     def __init__(self, data_path, patch_size = 11, random_seed=1234, k=5, m_list=[0,1,2,3,4]):
#           
#         total_data_size = 0
#         for f in range(1, 101):
#             with open(os.path.join(data_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'rb') as fp:
#                 dataset = pickle.load(fp)
#                 total_data_size += dataset['label'].shape[0]
# 
#         data = np.empty((total_data_size, patch_size*patch_size*3))
#         label = np.empty((total_data_size, 2))
# 
#         data_index = 0
#         for f in range(1, 101):
#             with open(os.path.join(data_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'rb') as fp:
#                 dataset = pickle.load(fp)
#                 data_size = dataset['label'].shape[0]
#                 data[data_index:(data_index+data_size), :] = (1.0/255.0)*dataset['data']
#                 label[data_index:(data_index+data_size), :] = dataset['label']
#                 data_index += data_size
#                      
#         sample_list_initial = range(total_data_size)
#         random.seed(random_seed)
#         random.shuffle(sample_list_initial)
#         sample_list_partitions = self.partition(sample_list_initial, k)
#         
#         sample_list = []
#         for m in m_list:
#             sample_list += sample_list_partitions[m]
#         
#         self.data = data[sample_list, :]
#         self.label = label[sample_list, :]
#                 
#     def __getitem__(self, index):
#         return self.data[index], self.label[index] 
#     
#     def __len__(self):
#         return self.label.shape[0]       
#             
#     def partition(self, lst, n):
#         division = len(lst) / float(n)
#         return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]                  
   
class LymphocyteDataset():
    def __init__(self, data_path, patch_size = 11, random_seed=1234, k=5, m_list=[0,1,2,3,4]):
        sample_list_initial = range(1, 101)
        random.seed(random_seed)
        random.shuffle(sample_list_initial)
        sample_list_partitions = self.partition(sample_list_initial, k)
              
        sample_list = []
        for m in m_list:
            sample_list += sample_list_partitions[m]
                              
        total_data_size = 0
        for f in sample_list:
            with open(os.path.join(data_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'rb') as fp:
                dataset = pickle.load(fp)
                total_data_size += dataset['label'].shape[0]

        data = np.empty((total_data_size, patch_size*patch_size*3))
        label = np.empty((total_data_size, 2))

        data_index = 0
        for f in sample_list:
            with open(os.path.join(data_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'rb') as fp:
                dataset = pickle.load(fp)
                data_size = dataset['label'].shape[0]
                data[data_index:(data_index+data_size), :] = (1.0/255.0)*dataset['data']
                label[data_index:(data_index+data_size), :] = dataset['label']
                data_index += data_size
        
        self.data = data
        self.label = label
                
    def __getitem__(self, index):
        return self.data[index], self.label[index] 
    
    def __len__(self):
        return self.label.shape[0]       
            
    def partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]  
    
def train_lymphocyte(output_file, input_folder, stage_id):
    '''Test the convolutional autoencder using lymphocyte.'''
    # %%
    
    lymphocyte_training_dataset = LymphocyteDataset(input_folder, patch_size=PATCH_SIZE, random_seed=RANDOM_SEED_MAGIC, k=10, m_list = [0,1,2,3,4,5,6,7,8])
    lymphocyte_test_dataset = LymphocyteDataset(input_folder, patch_size=PATCH_SIZE, random_seed=RANDOM_SEED_MAGIC, k=10, m_list = [9])    
    lymphocyte_data, _ = lymphocyte_training_dataset[random.sample(range(lymphocyte_training_dataset.__len__()), 1000)]
    mean = np.mean(lymphocyte_data, axis=0)

       
    xae_learning_rate = 0.00005
    fcn_learning_rate = 0.00001
    n_xae_batch_size = 1000
    n_fcn_batch_size = 1000
    n_xae_epochs = 1000000
    n_fcn_epochs = 1000000
    n_xae_reload_per_epochs = 100
    n_xae_display_per_epochs = 10000
    n_fcn_reload_per_epochs = 100
    n_fcn_display_per_epochs = 10000
    
    xae_layers = [         
        {
            'n_channels': 225,
            'reconstructive_regularizer': 1.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'corrupt_prob': 1.0,
            'tied_weight': True,
            'exclusive_type': 'logcosh',
            'exclusive_scale': 10.0,    
            # 'gaussian_mean': 0.0,    
            # 'gaussian_std': 0.0,             
            'encode':'bswish', 'decode':'linear',
            'pathways': [
                range(0, 150),
                range(75, 225),
            ],
        },                  
    ]
     
    xae = eXclusiveAutoencoder(
        input_dimensions = lymphocyte_data.shape[1],
        layers = xae_layers,
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,
    )
    
    fcn_dimensions = [225, 45, 9, 2]
    fcn = FullyConnectedNetwork(
        dimensions = fcn_dimensions,
        init_weight = None,
        init_bias = None,  
    )
    
    xae_optimizer_list = []  
     
    for layer_i in range(len(xae_layers)):
        xae_optimizer_list.append(AMSGrad.AMSGrad(xae_learning_rate).minimize(xae['layerwise_cost'][layer_i]['total'], var_list=[
                xae['encoder_weight'][layer_i],
                xae['encoder_bias'][layer_i],
                # xae['decoder_weight'][layer_i],
                # xae['decoder_bias'][layer_i],
        ]))
         
    xae_optimizer_list.append(AMSGrad.AMSGrad(xae_learning_rate).minimize(xae['cost']['total']))
    
    fcn_optimizer = tf.train.AdamOptimizer(fcn_learning_rate).minimize(fcn['cost'])
    # fcn_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(fcn['cost'])
    
    # fcn_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fc['cost'])
    
    # correct_prediction = tf.equal(tf.argmax(fcn['y'], 1), tf.argmax(fcn['y_'], 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # %%
    # We create a session to use the graph
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data
         
    for layer_i, (optimizer) in enumerate(xae_optimizer_list):
        for epoch_i in range(n_xae_epochs): 
            if (epoch_i) % n_xae_reload_per_epochs == 0:
                batch_xs, batch_ys = lymphocyte_training_dataset[random.sample(range(lymphocyte_training_dataset.__len__()), n_xae_batch_size)]
                train = []
                train.append(np.array([img - mean for img in batch_xs[np.where(np.any(np.array([
                    batch_ys[:, 0], 
                    # batch_ys[:, 1],                    
                ]) == 1, axis=0))]]))
                train.append(np.array([img - mean for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    batch_ys[:, 1],
                ]) == 1, axis=0))]]))
                     
            sess.run(optimizer, feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    
            if (epoch_i+1) % n_xae_display_per_epochs == 0:
                if optimizer is xae_optimizer_list[-1]:
                    cost_total = sess.run(xae['cost']['total'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_reconstruction_error = sess.run(xae['cost']['reconstruction_error'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_sparsity = sess.run(xae['cost']['sparsity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_exclusivity = sess.run(xae['cost']['exclusivity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_weight_decay = sess.run(xae['cost']['weight_decay'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                else:
                    cost_total = sess.run(xae['layerwise_cost'][layer_i]['total'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_reconstruction_error = sess.run(xae['layerwise_cost'][layer_i]['reconstruction_error'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_sparsity = sess.run(xae['layerwise_cost'][layer_i]['sparsity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_exclusivity = sess.run(xae['layerwise_cost'][layer_i]['exclusivity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    cost_weight_decay = sess.run(xae['layerwise_cost'][layer_i]['weight_decay'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                 
                print("layer: {}, epoch:{:6d}, cost: {:.5f}, error: {:.5f}, sparsity: {:.5f}, exclusivity: {:.5f}, weight decay: {:.5f}".format(
                    'A' if optimizer is xae_optimizer_list[-1] else layer_i+1, 
                    epoch_i+1, 
                    cost_total, 
                    cost_reconstruction_error, 
                    cost_sparsity, 
                    cost_exclusivity, 
                    cost_weight_decay))
                
                test_xs, test_ys = lymphocyte_training_dataset[random.sample(range(lymphocyte_training_dataset.__len__()), 10240)]
                
                test_xs_0 = np.array([img - mean for img in test_xs[np.where(np.any(np.array([
                    test_ys[:, 0], 
                    # test_ys[:, 1], 
                ]) == 1, axis=0))][:256]])
                   
                test_xs_1 = np.array([img - mean for img in test_xs[np.where(np.any(np.array([
                    # test_ys[:, 0], 
                    test_ys[:, 1], 
                ]) == 1, axis=0))][:256]])
                
                if optimizer is xae_optimizer_list[-1]:
                    recon_0 = sess.run(xae['y'], feed_dict={xae['x']: test_xs_0})
                    recon_1 = sess.run(xae['y'], feed_dict={xae['x']: test_xs_1})
                else:
                    recon_0 = sess.run(xae['layerwise_y'][layer_i], feed_dict={xae['x']: test_xs_0})
                    recon_1 = sess.run(xae['layerwise_y'][layer_i], feed_dict={xae['x']: test_xs_1})
                
                weights = sess.run(xae['encoder_weight'][0])
                display_color_network(weights, filename='lymphocyte_'+stage_id+'_weights.png')                             
                display_color_network(test_xs_0.transpose(), filename='lymphocyte_'+stage_id+'_test_0.png')
                display_color_network(recon_0.transpose(), filename='lymphocyte_'+stage_id+'_results_0.png')              
                display_color_network(test_xs_1.transpose(), filename='lymphocyte_'+stage_id+'_test_1.png')
                display_color_network(recon_1.transpose(), filename='lymphocyte_'+stage_id+'_results_1.png')              
 
    for epoch_i in range(n_fcn_epochs):
        if (epoch_i) % n_fcn_reload_per_epochs == 0:
            batch_xs, batch_ys = lymphocyte_training_dataset[random.sample(range(lymphocyte_training_dataset.__len__()), n_fcn_batch_size)]
            batch_xs = batch_xs-np.tile(mean, (batch_xs.shape[0], 1))
              
        ae_z = sess.run(xae['z'], feed_dict={xae['x']: batch_xs})
        
        sess.run(fcn_optimizer, feed_dict={fcn['x']: ae_z, fcn['y_']: batch_ys, fcn['keep_prob_input']: 0.9, fcn['keep_prob']: 0.9})
            
        if (epoch_i+1) % n_fcn_display_per_epochs == 0:
            cost = sess.run(fcn['cost'], feed_dict={fcn['x']: ae_z, fcn['y_']: batch_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
            
            batch_xs, batch_ys = lymphocyte_test_dataset[:]
            batch_xs = batch_xs-np.tile(mean, (batch_xs.shape[0], 1))
            ae_z = sess.run(xae['z'], feed_dict={xae['x']: batch_xs})
            
            acc = sess.run(fcn['accuracy'], feed_dict={fcn['x']: ae_z, fcn['y_']: batch_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
            
            print('FCN epoch:{:6d}, cost: {:.5f}, accuracy: {:.5f}'.format(epoch_i+1, cost, acc))
                 
    test_xs, test_ys = lymphocyte_test_dataset[:]
    test_xs = test_xs-np.tile(mean, (test_xs.shape[0], 1))
    ae_z = sess.run(xae['z'], feed_dict={xae['x']: test_xs})
    
    y = sess.run(fcn['y'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
    prediction = sess.run(fcn['prediction'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
    conf_matrix = confusion_matrix(test_ys[:, 1], prediction)

    print('final f1_score:', f1_score(test_ys[:, 1], prediction, average="macro"))
    print('final precision:', precision_score(test_ys[:, 1], prediction, average="macro"))
    print('final recall:', recall_score(test_ys[:, 1], prediction, average="macro"))
    print('final average_precision:', average_precision_score(test_ys[:, 1], prediction, average="macro"))                    
    print('final accuracy:', sess.run(fcn['accuracy'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0}))
                
    sys.stdout.write('\n\nConfusion Matrix')
    sys.stdout.write('\t'*(len(LABEL_LIST)-2)+'| Accuracy')
    sys.stdout.write('\n'+'-'*8*(len(LABEL_LIST)+1))
    sys.stdout.write('\n')
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            sys.stdout.write(str(conf_matrix[i][j].astype(np.int))+'\t')
        sys.stdout.write('| %3.2f %%' % (conf_matrix[i][i]*100 / conf_matrix[i].sum()))
        sys.stdout.write('\n')
    sys.stdout.write('Number of test samples: %i \n\n' % conf_matrix.sum())
             
            
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(test_ys[:,1], y[:,1])
    roc_auc = auc(fpr, tpr)
     
    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(test_ys.ravel(), y.ravel())
    roc_auc_micro = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class  
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    
    plt.gcf().savefig('lymphocyte_'+stage_id+'_rocplot.png')
    plt.clf()
        
    writer.close()
    
    xae_encoder_weight = sess.run(xae['encoder_weight'])
    xae_encoder_bias = sess.run(xae['encoder_bias'])
    xae_decoder_weight = sess.run(xae['decoder_weight'])
    xae_decoder_bias = sess.run(xae['decoder_bias'])
    fcn_weight = sess.run(fcn['weight'])
    fcn_bias = sess.run(fcn['bias'])
 
    data = {
        'xae_layers': xae_layers,
        'fcn_dimensions': fcn_dimensions,
        'xae_encoder_weight': xae_encoder_weight,
        'xae_encoder_bias': xae_encoder_bias,
        'xae_decoder_weight': xae_decoder_weight,
        'xae_decoder_bias': xae_decoder_bias,
        'fcn_weight': fcn_weight,
        'fcn_bias': fcn_bias,
        'mean': mean,
        'patch_size': PATCH_SIZE,
        }
     
    with open(output_file, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL) 
        
def test_lymphocyte():
    '''Test the convolutional autoencder using lymphocyte.'''
    # %%
       
    lymphocyte_test_dataset = LymphocyteDataset('./lymphocyte', patch_size=PATCH_SIZE, random_seed=RANDOM_SEED_MAGIC, k=10, m_list = [9])    

     
    with open('lymphocyte.pickle', 'rb') as fp:
        data = pickle.load(fp) 
     
    xae = eXclusiveAutoencoder(
        input_dimensions = data['mean'].shape[0],
        layers = data['xae_layers'],        
        init_encoder_weight = data['xae_encoder_weight'],
        init_decoder_weight = data['xae_decoder_weight'],
        init_encoder_bias = data['xae_encoder_bias'],
        init_decoder_bias = data['xae_decoder_bias'],
    )
    
    fcn = FullyConnectedNetwork(
        dimensions = data['fcn_dimensions'],
        init_weight = data['fcn_weight'],
        init_bias = data['fcn_bias'],  
    )
    
    # %%
    # We create a session to use the graph
    
    sess = tf.Session()
    writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data

    test_xs, test_ys = lymphocyte_test_dataset[:]
    ae_z = sess.run(xae['z'], feed_dict={xae['x']: test_xs})
    
    y = sess.run(fcn['y'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
    prediction = sess.run(fcn['prediction'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0})
    conf_matrix = confusion_matrix(test_ys[:, 1], prediction)
    
    print('final f1_score:', f1_score(test_ys[:, 1], prediction, average="macro"))
    print('final precision:', precision_score(test_ys[:, 1], prediction, average="macro"))
    print('final recall:', recall_score(test_ys[:, 1], prediction, average="macro"))
    print('final average_precision:', average_precision_score(test_ys[:, 1], prediction, average="macro"))                    
    print('final accuracy:', sess.run(fcn['accuracy'], feed_dict={fcn['x']: ae_z, fcn['y_']: test_ys, fcn['keep_prob_input']: 1.0, fcn['keep_prob']: 1.0}))
    
            
    print('\n\nConfusion Matrix')
    print('\t'*(len(LABEL_LIST)-2)+'| Accuracy')
    print('\n'+'-'*8*(len(LABEL_LIST)+1))
    print('\n')
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            print(str(conf_matrix[i][j].astype(np.int))+'\t')
        print('| %3.2f %%' % (conf_matrix[i][i]*100 / conf_matrix[i].sum()))
        print('\n')
    print('Number of test samples: %i \n\n' % conf_matrix.sum())
             
             
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(test_ys[:,1], y[:,1])
    roc_auc = auc(fpr, tpr)
     
    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(test_ys.ravel(), y.ravel())
    roc_auc_micro = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class  
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    
    plt.gcf().savefig('rocplot.png')
    plt.clf()
    writer.close()

        
# def lymphocyte_classification_dataset_preparation(result_path, raw_img_path, label_img_path, patch_size):
#     for f in range(1, 101):
#         data = np.empty((0, patch_size*patch_size*3))
#         label = np.empty((0, 2))  
#         
#         raw_img = np.array(Image.open(os.path.join(raw_img_path, 'im'+str(f)+'.tif')))
#         label_img = np.array(Image.open(os.path.join(label_img_path, str(f)+'m.tif')))[:, :, 1]/255.0
# 
#         for i in range(label_img.shape[0]-patch_size):
#             for j in range(label_img.shape[1]-patch_size):                                
#                 if label_img[i+int(patch_size/2), j+int(patch_size/2)] == 1.0:
#                     patch = raw_img[i:i+patch_size, j:j+patch_size].transpose((2, 0, 1))
#                     data = np.concatenate((data,
#                         patch[:, ::1, ::1].reshape((1, patch_size*patch_size*3)),
#                         patch[:, ::-1, ::1].reshape((1, patch_size*patch_size*3)),
#                         patch[:, ::1, ::-1].reshape((1, patch_size*patch_size*3)),
#                         patch[:, ::-1, ::-1].reshape((1, patch_size*patch_size*3)),                                                
#                         ), axis=0)
#                     label = np.concatenate((label, np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]])), axis=0)
#                 elif label_img[i:i+patch_size, j:j+patch_size].max() == 0.0:
#                     if np.random.rand() < 4*0.0048*20:
#                         patch = raw_img[i:i+patch_size, j:j+patch_size].transpose((2, 0, 1))
#                         
#                         data = np.concatenate((data,
#                             patch.reshape((1, patch_size*patch_size*3)),                                              
#                             ), axis=0)
#                                                 
#                         label = np.concatenate((label, np.array([[1.0, 0.0]])), axis=0)
#               
#         dataset = {
#             'data': data,
#             'label': label,
#             }
#      
#         with open(os.path.join(result_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'wb') as fp:
#             pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
# def lymphocyte_detection_dataset_preparation(result_path, raw_img_path, label_img_path, patch_size):
#     for f in range(1, 101):
#         data = np.empty((0, patch_size*patch_size*3))
#         label = np.empty((0, 2))  
#          
#         raw_img = np.array(Image.open(os.path.join(raw_img_path, 'im'+str(f).zfill(3)+'.tif')))
#         label_img = np.array(Image.open(os.path.join(label_img_path, str(f).zfill(3)+'m.tif')))[:, :, 1]/255.0
#  
#         pos = []
#         theta = []
#         scale = []
#         flip = []
#         
#         label = []
#         
#         for i in range(label_img.shape[0]-patch_size):
#             for j in range(label_img.shape[1]-patch_size):                                
#                 if label_img[i+int(patch_size/2), j+int(patch_size/2)] == 1.0:
#                     for k in range(DATA_AUGMENTATION):
#                         pos.append([
#                             j+int(patch_size/2), 
#                             i+int(patch_size/2),
#                             ])
#                         theta.append(np.random.rand()*2*math.pi)
#                         scale.append(1.0)
#                         flip.append(np.random.randint(2)==0)
#                         label.append([0.0, 1.0])
#                 elif label_img[i:i+patch_size, j:j+patch_size].max() == 0.0:
#                     if np.random.rand() < 0.0048*DATA_AUGMENTATION:
#                         pos.append([
#                             j+int(patch_size/2), 
#                             i+int(patch_size/2),
#                             ])
#                         theta.append(0.0)
#                         scale.append(1.0)
#                         flip.append(np.random.randint(2)==0)
#                         label.append([1.0, 0.0])
#            
#         sample_ch0 = pcs.sampling(raw_img[:, :, 0].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool)) 
#         sample_ch1 = pcs.sampling(raw_img[:, :, 1].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool)) 
#         sample_ch2 = pcs.sampling(raw_img[:, :, 2].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool))
#         
#         dataset = {
#             'data': np.hstack((sample_ch0, sample_ch1, sample_ch2)),
#             'label': np.array(label),
#             }
#       
#         with open(os.path.join(result_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'wb') as fp:
#             pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         
def lymphocyte_classification_dataset_preparation(result_path, csv_file, raw_img_path, label_img_path, patch_size):
    cnt_pos = 0
    cnt_neg = 0
     
    for f in range(1, 101):
        data = np.empty((0, patch_size*patch_size*3))
        label = np.empty((0, 2))  
          
        raw_img = np.array(Image.open(os.path.join(raw_img_path, 'im'+str(f).zfill(3)+'.tif')))
        label_img = np.array(Image.open(os.path.join(label_img_path, str(f).zfill(3)+'m.tif')))[:, :, 1]/255.0
  
        pos = []
        theta = []
        scale = []
        flip = []
         
        label = []
         
        for i in range(label_img.shape[0]-patch_size+1):
            for j in range(label_img.shape[1]-patch_size+1):                                
                if label_img[i+int(patch_size/2), j+int(patch_size/2)] == 1.0:
                    for k in range(int(round(2.55*10))):
                        cnt_pos += 1
                        pos.append([
                            j+int(patch_size/2), 
                            i+int(patch_size/2),
                            ])
                        theta.append(np.random.rand()*2*math.pi)
                        scale.append(1.0)
                        flip.append(np.random.randint(2)==0)
                        label.append([0.0, 1.0])
 
        with open(csv_file) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                imgNum = int(row['ImageNumber'])
                 
                if imgNum == f:        
                    # objNum = int(row['ObjectNumber'])
                    j, i = int(round(float(row['Location_Center_X']))), int(round(float(row['Location_Center_Y'])))    
                    # print(imgNum, objNum, x, y)
                    if label_img[i-int(patch_size/2):i+int(patch_size/2)+1, j-int(patch_size/2):j+int(patch_size/2)+1].max() == 0.0:
                        for k in range(int(round(4.21*10))):
                            cnt_neg += 1
                            pos.append([
                                j, 
                                i,
                                ])
                            theta.append(np.random.rand()*2*math.pi)
                            scale.append(1.0)
                            flip.append(np.random.randint(2)==0)
                            label.append([1.0, 0.0])
                         
            
        sample_ch0 = pcs.sampling(raw_img[:, :, 0].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool)) 
        sample_ch1 = pcs.sampling(raw_img[:, :, 1].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool)) 
        sample_ch2 = pcs.sampling(raw_img[:, :, 2].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(flip).astype(np.bool))
         
        dataset = {
            'data': np.hstack((sample_ch0, sample_ch1, sample_ch2)),
            'label': np.array(label),
            }
       
        with open(os.path.join(result_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'wb') as fp:
            pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
       
def lymphocyte_detection_dataset_preparation(result_path, raw_img_path, label_img_path, patch_size):
    samples_num  = [0, 0]
    
    for f in range(1, 101):
        data = np.empty((0, patch_size*patch_size*3))
        label = np.empty((0, 2))  
         
        raw_img = np.array(Image.open(os.path.join(raw_img_path, 'im'+str(f).zfill(3)+'.tif')))
        label_img = np.array(Image.open(os.path.join(label_img_path, str(f).zfill(3)+'m.tif')))[:, :, 1]/255.0
 
        pos = []
        theta = []
        scale = []
        flip = []
        label = []
        
        for i in range(label_img.shape[0]-patch_size+1):
            for j in range(label_img.shape[1]-patch_size+1):                                
                if label_img[i+int(patch_size/2), j+int(patch_size/2)] == 1.0:
                    for k in range(65*DATA_AUGMENTATION):
                        samples_num[0] += 1
                        pos.append([
                            j+int(patch_size/2), 
                            i+int(patch_size/2),
                            ])
                        theta.append(np.random.rand()*2*math.pi)
                        scale.append(1.0)
                        flip.append(np.random.randint(2)==0)
                        label.append([0.0, 1.0])
                elif label_img[i:i+patch_size, j:j+patch_size].max() == 1.0 and label_img[i+2:i+patch_size-2, j+2:j+patch_size-2].max() == 0.0:
                    # if np.random.rand() < 0.0048*DATA_AUGMENTATION:
                    # if np.random.rand() < 1:
                    for _ in range(DATA_AUGMENTATION):
                        samples_num[1] += 1
                        pos.append([
                            j+int(patch_size/2), 
                            i+int(patch_size/2),
                            ])
                        theta.append(0.0)
                        scale.append(1.0)
                        flip.append(np.random.randint(2)==0)
                        label.append([1.0, 0.0])                        
#                 elif label_img[i:i+patch_size, j:j+patch_size].max() == 0.0:
#                     if np.random.rand() < 0.0048*DATA_AUGMENTATION:
#                         pos.append([
#                             j+int(patch_size/2), 
#                             i+int(patch_size/2),
#                             ])
#                         theta.append(0.0)
#                         scale.append(1.0)
#                         flip.append(np.random.randint(2)==0)
#                         label.append([1.0, 0.0])
           
        sample_ch0 = pcs.sampling(raw_img[:, :, 0].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(scale).astype(np.bool)) 
        sample_ch1 = pcs.sampling(raw_img[:, :, 1].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(scale).astype(np.bool)) 
        sample_ch2 = pcs.sampling(raw_img[:, :, 2].astype(np.float32), patch_size, np.array(pos).astype(np.int32), np.array(theta).astype(np.float32), np.array(scale).astype(np.float32), np.array(scale).astype(np.bool))
        
        dataset = {
            'data': np.hstack((sample_ch0, sample_ch1, sample_ch2)),
            'label': np.array(label),
            }
      
        with open(os.path.join(result_path, 'lymphocyte_dataset_'+str(f)+'.pickle'), 'wb') as fp:
            pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(samples_num)
         
if __name__ == '__main__':
#     lymphocyte_detection_dataset_preparation(
#         result_path='./dataset/detection', 
#         raw_img_path='./dataset/normalized_data', 
#         label_img_path='./dataset/manual_seg/center', 
#         patch_size=PATCH_SIZE)
#           
#     train_lymphocyte(output_file='lymphocyte_detection.pickle', 
#                      input_folder='./dataset/detection',
#                      stage_id='stage1')
#      
#     lymphocyte_classification_dataset_preparation(
#         result_path='./dataset/classification', 
#         csv_file='./lymphocyte_IdentifyPrimaryObjects1.csv', 
#         raw_img_path='./dataset/normalized_data',
#         label_img_path='./dataset/manual_seg/center', 
#         patch_size=PATCH_SIZE)
     
    train_lymphocyte(
        output_file='./lymphocyte_classification.pickle', 
        input_folder='./dataset/classification',
        stage_id='classification')
    
    