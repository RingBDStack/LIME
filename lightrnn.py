# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pdb
import random
import math
import sys

import numpy as np
import tensorflow as tf

class LightRNN(object):
	def __init__(self, opt, reuse=None):
		with tf.variable_scope("{}_input".format(opt.mode)):
			# Input scope should not be shared between models
			#assert tf.get_variable_scope().reuse == False	
			#with tf.device("/cpu:0"):  # Place queue on parameter server.
				self.x_r = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="x_r")
				self.x_c = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="x_c")
				self.y_r = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="y_r")
				self.y_c = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="y_c")
				self.y = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="y")
				
				self.data_queue = tf.FIFOQueue(capacity=opt.lightrnn_size, dtypes=[tf.int32, tf.int32, tf.int32, tf.int32, tf.int32], shapes=[[opt.num_steps, opt.batch_size], [opt.num_steps, opt.batch_size], [opt.num_steps, opt.batch_size], [opt.num_steps, opt.batch_size], [opt.num_steps, opt.batch_size]], shared_name="{}_shared_queue".format(opt.mode), name="{}_queue".format(opt.mode))
				
				self.enqueue_op = self.data_queue.enqueue([self.x_r, self.x_c, self.y_r, self.y_c, self.y])
				queue_outputs = self.data_queue.dequeue()

		with tf.variable_scope("model", reuse=reuse), tf.name_scope("{}_model".format(opt.mode)):
			if opt.mode == "predict":
				self.data_r = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="data_r")
				self.data_c = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="data_c")
				self.target_r = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="target_r")
				self.target_c = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="target_c")
				self.target = tf.placeholder(dtype=tf.int32, shape=[opt.num_steps, opt.batch_size], name="target")
			else:
				self.data_r, self.data_c, self.target_r, self.target_c, self.target = queue_outputs
			
			with tf.name_scope("embedding"):
				stdv = np.sqrt(1. / opt.lightrnn_size)
				self.embedding_r = tf.get_variable("embedding_r", [opt.lightrnn_size, opt.embedding_size], initializer=tf.random_uniform_initializer(-stdv, stdv))
				self.embedding_c = tf.get_variable("embedding_c", [opt.lightrnn_size, opt.embedding_size], initializer=tf.random_uniform_initializer(-stdv, stdv))

			input_r = tf.nn.embedding_lookup(self.embedding_r, self.data_r)
			input_c = tf.nn.embedding_lookup(self.embedding_c, self.data_c)
			input_r = tf.nn.dropout(input_r, opt.input_keep_prob) 
			input_c = tf.nn.dropout(input_c, opt.input_keep_prob) 
			
			def lstm_cell():
				raw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(opt.hidden_size, reuse=tf.get_variable_scope().reuse)
				return tf.contrib.rnn.DropoutWrapper(raw_lstm_cell, output_keep_prob=opt.lstm_keep_prob)
			
			cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(opt.num_layers)])
			self.initial_state = cell.zero_state(opt.batch_size, dtype=tf.float32)
			state_c = self.initial_state
		
			softmax_w_r = tf.get_variable("softmax_w_r", [opt.hidden_size, opt.lightrnn_size])
			softmax_b_r = tf.get_variable("softmax_b_r", [opt.lightrnn_size])
			softmax_w_c = tf.get_variable("softmax_w_c", [opt.hidden_size, opt.lightrnn_size])
			softmax_b_c = tf.get_variable("softmax_b_c", [opt.lightrnn_size])
		
			cell_outputs_r = []
			cell_outputs_c = []
			with tf.variable_scope("RNN"):
				for time_step in range(opt.num_steps):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					
					cell_output_c, state_r = cell(input_r[time_step], state_c)
					
					tf.get_variable_scope().reuse_variables()
					cell_output_r, state_c = cell(input_c[time_step], state_r)
					outputs_r = tf.matmul(cell_output_r, softmax_w_r) + softmax_b_r
					
					if opt.mode == "train":
						"""	
						# When training, randomly mix ground true label with our previously predicted label
						true_input_data_rc = self.target_r[time_step]
						true_inputs_rc = tf.nn.embedding_lookup(embedding_r, true_input_data_rc)
					
						norm_outputs_r = tf.nn.softmax(outputs_r)
						my_inputs_rc = tf.matmul(norm_outputs_r, embedding_r)
					
						# Here we set the ratio of ground true one hot vector
						inputs_rc = tf.where(tf.random_uniform([]) < opt.input_rc_ratio, true_inputs_rc, my_inputs_rc)
						"""
						true_input_data_rc = self.target_r[time_step]
						inputs_rc = tf.nn.embedding_lookup(self.embedding_r, true_input_data_rc)
						
					elif opt.mode == "valid":
						
						true_input_data_rc = self.target_r[time_step]
						inputs_rc = tf.nn.embedding_lookup(self.embedding_r, true_input_data_rc)
					
					elif opt.mode == "test" or opt.mode == "predict":
						# Use every r in range(vocab_size) to predict c
						all_outputs_r = tf.constant(np.tile(np.arange(opt.lightrnn_size, dtype=np.int32), (opt.batch_size,1)))
						inputs_rc = tf.nn.embedding_lookup(self.embedding_r, all_outputs_r)

					# Does this need to be dropout again???
					#inputs_rc = tf.nn.dropout(inputs_rc, self.input_keep_prob)

					if opt.mode == "test" or opt.mode == "predict":
						cell_outputs_c_list = []
						for r in range(opt.lightrnn_size):
							tf.get_variable_scope().reuse_variables()
							cell_output_c, _ = cell(inputs_rc[:,r,:], state_c)
							cell_outputs_c_list.append(cell_output_c)
						cell_outputs_c_tensor = tf.reshape(tf.concat(cell_outputs_c_list, axis=1), [-1, opt.hidden_size])
						outputs_c = tf.reshape(tf.matmul(cell_outputs_c_tensor, softmax_w_c) + softmax_b_c, [opt.batch_size, opt.vocab_size])
					else:
						tf.get_variable_scope().reuse_variables()
						cell_output_c, state_r = cell(inputs_rc, state_c)
						outputs_c = tf.matmul(cell_output_c, softmax_w_c) + softmax_b_c
					
					cell_outputs_r.append(outputs_r)
					cell_outputs_c.append(outputs_c)
			
			# Evaluate model
			# The followings are all time-majored, data within one timestep are bind together, 
			logits_r = tf.concat(cell_outputs_r, axis=0)
			logits_c = tf.concat(cell_outputs_c, axis=0)
			output_prob_r = tf.nn.softmax(logits_r)
			output_prob_c = tf.nn.softmax(logits_c)
			self.output_loss_r = -tf.nn.log_softmax(logits_r)
			self.output_loss_c = -tf.nn.log_softmax(logits_c)
				
			with tf.name_scope("loss"):
				loss_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
								 labels = tf.reshape(self.target_r, [opt.batch_size*opt.num_steps]),
								 logits = logits_r,
								 name = "loss_r"))
				loss_c = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
								 labels = tf.reshape(self.target_c, [opt.batch_size*opt.num_steps]),
								 logits = logits_c,
								 name = "loss_c"))
				self.loss = loss_r + loss_c
				# create a summary for our losses
				
			with tf.name_scope("learning_rate"):
				self.lr = tf.get_variable('lr', [], initializer=tf.constant_initializer(opt.initial_lr), trainable=False)
				self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
				self.lr_decay_op = self.lr.assign(self.new_lr)
				self.lr_init_op = self.lr.assign(opt.initial_lr)
			
			if opt.mode == "train":
				with tf.name_scope("train"):
					if opt.use_adam:	
						optimizer = tf.train.AdamOptimizer(use_locking=True) # Adam Optimizer
					else:	
						optimzer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
	
					tvars = tf.trainable_variables()
					grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), opt.max_grad_norm)
					self.train_op = optimizer.apply_gradients(zip(grads, tvars))
			
			if opt.mode == "test" or opt.mode == "predict": 
				with tf.name_scope("prob"):
					prob_r = tf.reshape(output_prob_r, [opt.num_steps*opt.batch_size, opt.lightrnn_size, 1])
					prob_r = tf.tile(prob_r, [1, 1, opt.lightrnn_size])
					prob_r = tf.reshape(prob_r, [opt.num_steps*opt.batch_size, -1])

					prob_c = output_prob_c
					prob = tf.multiply(prob_r, prob_c)
		
				with tf.name_scope("predict"):	
					_, self.pred_topK = tf.nn.top_k(prob, opt.top_num)
			
				with tf.name_scope("accuracy"):
					top_k=tf.nn.in_top_k(prob, tf.reshape(self.target, [-1]), opt.top_num)  
					self.accuracy = tf.reduce_mean(tf.cast(top_k, tf.float32))

	def update_lr(self, sess, new_lr):
		sess.run(self.lr_decay_op, feed_dict={self.new_lr: new_lr})	


