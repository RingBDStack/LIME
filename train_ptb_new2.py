# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import pdb
import random
import math
import sys
import pickle

import threading
import numpy as np
import tensorflow as tf
from ortools.graph import pywrapgraph
from lightrnn import LightRNN
from data_util import Reader

flags = tf.app.flags
logging = tf.logging

flags.DEFINE_string("data_path", "/home/LAB/penghao/nyx/HIN/LightRNN/data", "data_path")
flags.DEFINE_string("dataset", "meta_path7", "The dataset we use for training")
flags.DEFINE_string("model_dir", "./model", "model_path")
flags.DEFINE_string("log_dir", "./log", "log_path")
flags.DEFINE_string("model_name", "lightRNN--mcmf-model", "model_path")

# Flags for defining the tf.train.ClusterSpec
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_integer('num_layers', 1, 'Number of layers in RNN')
flags.DEFINE_integer('num_steps', 5, 'Number of steps for BPTT')
flags.DEFINE_integer('embedding_size', 200, 'Embedding dimension for one word')
flags.DEFINE_integer('hidden_size', 512, 'Number of hidden nodes for one layer')
flags.DEFINE_integer('max_adjust_iters', 4, 'Number of dictionary adjustion before stop')
flags.DEFINE_integer('batch_size', 256, 'Number of lines in one batch for training')
flags.DEFINE_integer('vocab_size', 252004, 'Size of vocabulary')
flags.DEFINE_integer('lightrnn_size', 502, 'Size of row and column vector to represent the word')
flags.DEFINE_integer('thread_num', 1, 'num of thread per queue')
flags.DEFINE_integer('train_valid_ratio', 9, 'ratio of size of train data and valid data, better be multiple of thread_num')
flags.DEFINE_integer('top_num', 3, 'num of top candidates when calculate accuracy')
flags.DEFINE_float("lr_decay_factor", 0.8, "The decay factor for learning rate")
flags.DEFINE_float("initial_lr", 1.0, "The initial learning rate for training model")
flags.DEFINE_float("lstm_keep_prob", 0.5, "The keep rate for lstm layers")
flags.DEFINE_float("input_keep_prob", 0.8, "The keep rate for input layer")
flags.DEFINE_float("input_rc_ratio", 1.0, "The ratio for ground true input_rc")
flags.DEFINE_float("max_grad_norm", 5.0, "The max norm that clip the gradients")
flags.DEFINE_bool("use_adam", True, "Use AdamOptimizer as training optimizer")
flags.DEFINE_bool("use_mcmf", True, "Use MCMF algorithm to adjust dictionary")

flags.DEFINE_bool("restore", False, "Restore model")
flags.DEFINE_bool("restore_rc", False, "Restore rc.pkl")
flags.DEFINE_bool("restart_after_adjustion", False, "initialize all params after every adjustion")
FLAGS = flags.FLAGS


class Option(object):
	def __init__(self, mode):
		self.mode = mode
		self.num_layers = FLAGS.num_layers
		self.embedding_size = FLAGS.embedding_size	
		self.hidden_size = FLAGS.hidden_size
		self.vocab_size = FLAGS.vocab_size
		self.lightrnn_size = FLAGS.lightrnn_size
		self.top_num = FLAGS.top_num
		self.initial_lr = FLAGS.initial_lr
		self.max_grad_norm = FLAGS.max_grad_norm
		self.use_adam = FLAGS.use_adam
		self.batch_size = FLAGS.batch_size
		self.num_steps = FLAGS.num_steps
		self.lstm_keep_prob = FLAGS.lstm_keep_prob if self.mode == "train" else 1.0
		self.input_keep_prob = FLAGS.input_keep_prob if self.mode == "train" else 1.0
		self.input_rc_ratio = FLAGS.input_rc_ratio


class LockedGen(object):
	def __init__(self, it):
		self.lock = threading.Lock()
		self.it = it.__iter__()

	def __iter__(self): 
		return self

	def next(self):
		self.lock.acquire()
		try:
			return self.it.__next__()
		finally:
			self.lock.release()


def split_train_valid_data(train_valid_data):
	train_valid_step = len(train_valid_data)
	train_step = train_valid_step // (FLAGS.train_valid_ratio + 1) * FLAGS.train_valid_ratio
	valid_step = train_valid_step - train_step
	valid_index = random.sample(range(train_valid_step), valid_step)
	train_index = [ind for ind in range(train_valid_step) if ind not in valid_index]
	
	train_data = [train_valid_data[i] for i in range(train_valid_step) if i not in valid_index]
	valid_data = [train_valid_data[i] for i in range(train_valid_step) if i in valid_index]
	
	return train_data, train_step, valid_data, valid_step	
		
def start_threads_func(reader, sess, coord):
	
	def feed_queue_data(data_gen, model, sess, coord):
		while not coord.should_stop():
			try:
				data_x_r, data_x_c, data_y_r, data_y_c, data_y = data_gen.next()
				sess.run(model.enqueue_op, feed_dict={	model.x_r:data_x_r, 
																								model.x_c:data_x_c, 
																								model.y_r:data_y_r, 
																								model.y_c:data_y_c, 
																								model.y:data_y	})
			except StopIteration:
				# Data finished for one epoch
				coord.request_stop()
				break
	
	def start_threads(data, model, thread_num):
		# Create fresh generator every time you call start_threads
		data_gen = LockedGen(reader.get_next_batch(data))
		threads = []
		for i in range(thread_num):
			t = threading.Thread(target=feed_queue_data, args=(data_gen, model, sess, coord))
			threads.append(t)
		for thread in threads:
			thread.daemon = True
			thread.start()
		return threads

	return start_threads

def main(_):
				
	# Create a cluster from the parameter server and worker hosts.
	
	# Start a server for a specific task
	
	#print("I'm worker %d and my server target is, "%FLAGS.task_index, server.target)	
			# Create reader for reading raw data	
	reader = Reader(FLAGS.data_path, FLAGS.dataset, FLAGS.vocab_size, FLAGS.batch_size, FLAGS.num_steps)
			
			# Create data from reader
	train_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.train.txt" % FLAGS.dataset)
	test_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.test.txt" % FLAGS.dataset)
	train_valid_data, _ = reader.read_file(train_path)
	test_data, test_step = reader.read_file(test_path)
		
		# Create options for each model
	train_opt = Option("train")
	valid_opt = Option("valid")
	test_opt = Option("test")
		
	train_model = LightRNN(train_opt, reuse=False)
	valid_model = LightRNN(valid_opt, reuse=True)
	test_model = LightRNN(test_opt, reuse=True)
		
	with tf.variable_scope("loss_matrix"):
		loss_matrix_r = tf.get_variable("loss_matrix_r", [FLAGS.vocab_size, FLAGS.lightrnn_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
		loss_matrix_c = tf.get_variable("loss_matrix_c", [FLAGS.vocab_size, FLAGS.lightrnn_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
				
		loss_matrix_update_r = tf.scatter_add(loss_matrix_r, tf.reshape(train_model.target, [-1]), train_model.output_loss_r, use_locking=True)
		loss_matrix_update_c = tf.scatter_add(loss_matrix_c, tf.reshape(train_model.target, [-1]), train_model.output_loss_c, use_locking=True)
		loss_matrix_update_op = tf.group(loss_matrix_update_r, loss_matrix_update_c)
			
	with tf.variable_scope("helper"):
		# Define training variables and ops
		adjustion = tf.get_variable("adjustion", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False)
		pre_adjustion = tf.get_variable("pre_adjustion", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False)
		do_adjustion = adjustion.assign(True)
		do_pre_adjustion = pre_adjustion.assign(True)
		epoch = tf.get_variable("epoch", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
		increment_epoch = tf.assign_add(epoch, 1)
		
	helper_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="helper")
	loss_matrix_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="loss_matrix")
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_vars)
	is_chief = 1
	with tf.Session() as sess:
			# Create a FileWriter to write summaries
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
			
		sess.run(tf.global_variables_initializer())		
		print("Variables initialized ...")
			
		if is_chief:
			# Create coordinator to control threads
			coord = tf.train.Coordinator()
					
			# Create start threads function
			start_threads = start_threads_func(reader, sess, coord)
		
		if FLAGS.restore:
			ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
			if ckpt_path:
				model_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
				print("Read model parameters from %s" % tf.train.latest_checkpoint(FLAGS.model_dir))
			else:
				print("model doesn't exists")	
			
		if is_chief and FLAGS.restore_rc:
			# Load wordid2r and wordid2c to update reader
			wordid2rc_path = os.path.join(FLAGS.model_dir, "wordid2rc.pkl")
			if os.path.isfile(wordid2rc_path):
				with open(wordid2rc_path, 'rb') as wordid2rc_file:
					reader.wordid2r = pickle.load(wordid2rc_file) 
					reader.wordid2c = pickle.load(wordid2rc_file)
			
			# Count step num for summary
		global_epoch = 0

		for adjust_iter in range(FLAGS.max_adjust_iters):
			print("start training with new wordid2id")
				
			if adjust_iter > 0:	
				sess.run(tf.variables_initializer(helper_vars))
				sess.run(tf.variables_initializer(loss_matrix_vars))
				if FLAGS.restart_after_adjustion:	
					sess.run(tf.variables_initializer(model_vars))
					
			if is_chief:
				# Randomly partition data into train and valid set
				train_data, train_step, valid_data, valid_step = split_train_valid_data(train_valid_data)
				
			ppl_history = []
			current_epoch = 0
			print("%s"%(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
			print ("train_step:%d, valid_step:%d, test_step:%d"%(train_step, valid_step, test_step))
			while not adjustion.eval():
				if is_chief:
					threads = start_threads(train_data, train_model, FLAGS.thread_num)
				if not pre_adjustion.eval():
					print ("global_epoch:%d"%(global_epoch))
					loss = 0.0
					step = 0
					start_time = time.time()
					#pdb.set_trace()
					run_options = tf.RunOptions(timeout_in_ms=1000)	
					while current_epoch == epoch.eval():
						try:
							train_model.update_lr(sess, FLAGS.initial_lr)
							loss_val, _ = sess.run([train_model.loss, train_model.train_op], options=run_options)
							loss += loss_val
							step += 1
							if step % 1000 == 0:
								print("	training step {}".format(step))
						except tf.errors.DeadlineExceededError:
							if is_chief and coord.should_stop():
								coord.join(threads, stop_grace_period_secs=10)
								coord.clear_stop()
								sess.run(increment_epoch)
							pass
					
					speed = step*FLAGS.num_steps*FLAGS.batch_size // (time.time()-start_time)
					train_ppl = np.exp(loss/step)
					print("	Train epoch: {} Train-PPL: {:.2f} step: {}  speed: {} wps worker_id: {}".format(current_epoch//2, train_ppl, step, speed, FLAGS.task_index))
				
				else:
					run_options = tf.RunOptions(timeout_in_ms=500)	
					while current_epoch == epoch.eval():
						try:
							sess.run(loss_matrix_update_op, options=run_options)	 
						except tf.errors.DeadlineExceededError:
							if is_chief and coord.should_stop():
								coord.join(threads, stop_grace_period_secs=10)
								coord.clear_stop()
								sess.run(increment_epoch)
							pass
				
				current_epoch += 1

				if is_chief:
					if not pre_adjustion.eval():
						loss = 0.0
						step = 0
						threads = start_threads(valid_data, valid_model, 1)
						while step < valid_step:
							loss_val = sess.run(valid_model.loss)
							loss += loss_val
							step += 1
						valid_ppl=np.exp(loss/step)		
						print("	Valid epoch: {} Valid-PPL: {:.2f} step: {}".format(current_epoch//2, valid_ppl, step))
						coord.join(threads, stop_grace_period_secs=10)
						coord.clear_stop()
				
						step = 0
						acc_list = []
						threads = start_threads(test_data, test_model, 1)
						while step < test_step:
							step_acc = sess.run(test_model.accuracy)
							acc_list.append(step_acc)
							step += 1
						test_acc = sum(acc_list)/len(acc_list)
						print("	Test epoch: {} Test-Accuracy: {:.4f} step: {}".format(current_epoch//2, test_acc, step))
						coord.join(threads, stop_grace_period_secs=10)
						coord.clear_stop()
						
							# If valid data performs bad, decay learning rate
						current_lr = train_model.lr.eval()
						if not FLAGS.use_adam and current_lr > 0.005 and len(ppl_history) > 0 and valid_ppl > ppl_history[-1]:
							current_lr *= FLAGS.lr_decay_factor
							train_model.update_lr(sess, current_lr)
						
							# If converged, do dictionary adjustion
							#if ((global_epoch+1)%5) == 0:
							#if current_epoch > 0 and current_epoch % 2 == 0:
						if((len(ppl_history) >= 3 and valid_ppl > max(ppl_history[-2:])) or (global_epoch+1) % 10 == 0):
						#if len(ppl_history) >= 2 and global_valid_ppl.eval() > ppl_history[-1]:
							sess.run(do_pre_adjustion)
						
						ppl_history.append(valid_ppl)
						global_epoch += 1
							
					else:
						_loss_matrix_r = loss_matrix_r.eval()
						_loss_matrix_c = loss_matrix_c.eval()
							
						print("Saving %d epoch model..."%(global_epoch-1))
							# Save graph and model parameters
						model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
						model_saver.save(sess, model_path, global_step=global_epoch-1)
							# Save wordid2r and wordid2c in reader
						wordid2rc_path = os.path.join(FLAGS.model_dir, "wordid2rc.pkl")
						with open(wordid2rc_path, 'wb') as wordid2rc_file:
							pickle.dump(reader.wordid2r, wordid2rc_file)
							pickle.dump(reader.wordid2c, wordid2rc_file)
							
						sess.run(do_adjustion)	
					sess.run(increment_epoch)
				else:
					while current_epoch == epoch.eval():
						pass
				current_epoch += 1
			embedding_r,embedding_c = sess.run([train_model.embedding_r,train_model.embedding_c])
			print("embedding shape is ",embedding_r.shape)
                                #print("embedding_r is ",embedding_r)
			np.save("embedding_r",embedding_r)
			np.save("embedding_c",embedding_c)
			
			if is_chief and adjust_iter < (FLAGS.max_adjust_iters - 1):
				print("	start adjusting...")
				original_total_cost = 0
				for i in range(FLAGS.vocab_size):
					original_total_cost += (_loss_matrix_r[i][i//FLAGS.lightrnn_size]+_loss_matrix_c[i][i%FLAGS.lightrnn_size])
				loss_matrix_r_file = open('probx.txt', 'w')
				for i in range(FLAGS.vocab_size):
					for j in range(FLAGS.lightrnn_size):
						loss_matrix_r_file.write("%f "%_loss_matrix_r[i][j])
					loss_matrix_r_file.write("\n")
				loss_matrix_r_file.close()
				loss_matrix_c_file = open('proby.txt', 'w')
				for i in range(FLAGS.vocab_size):
					for j in range(FLAGS.lightrnn_size):
						loss_matrix_c_file.write("%f "%_loss_matrix_c[i][j])
					loss_matrix_c_file.write("\n")
				loss_matrix_c_file.close()
				start_time = time.time()
				idx2wordfile = FLAGS.dataset+"_idx2word.txt"
				mapxyfile = FLAGS.dataset+"mcmf_mapxy.txt"
				if FLAGS.use_mcmf:
					adjust_dic_c="./a.out probx.txt proby.txt "+mapxyfile+" "+idx2wordfile+" "+ str(global_epoch-1)
				else:
					adjust_dic_c="./implus probx.txt proby.txt "+mapxyfile+" "+idx2wordfile+" "+ str(global_epoch-1)
				rc,out=subprocess.getstatusoutput(adjust_dic_c)
				total_adjustion = 0
				after_adjust_cost = 0
				rc = 0
				if (rc==0):
					print ("	open mapxy.txt")	
					mapxy_file=open(mapxyfile,"r")
					id2wordid = np.zeros(FLAGS.vocab_size, dtype=np.int32)
					for i in range(FLAGS.vocab_size):
						id2wordid[reader.wordid2r[i] * FLAGS.lightrnn_size + reader.wordid2c[i]] = i
					after_adjust_cost = 0
					count_temp=0
					idx2wordfile = FLAGS.dataset+"_idx2word.txt"
					idx2word_file = open(idx2wordfile, "w")
					for line in mapxy_file:
						line_val=line.split()
						for i in range(len(line_val)):
							orgin_id = int(line_val[i])-1
							true_id = id2wordid[orgin_id]
							idx2word_file.write(reader.vocab.words[true_id])
							idx2word_file.write("\n")
							after_adjust_cost += (_loss_matrix_r[orgin_id][count_temp // FLAGS.lightrnn_size]+_loss_matrix_c[orgin_id][count_temp % FLAGS.lightrnn_size])
								#true_id = id2wordid[count_temp]
							reader.wordid2r[true_id] = count_temp // FLAGS.lightrnn_size
							#reader.wordid2r[true_id] = (int(line_val[i])-1) // FLAGS.lightrnn_size
							reader.wordid2c[true_id] = count_temp % FLAGS.lightrnn_size
								#reader.wordid2c[true_id] = (int(line_val[i])-1) % FLAGS.lightrnn_size
							if orgin_id != count_temp:
								total_adjustion += 1
							count_temp=count_temp+1
					idx2word_file.close()
				else:
					print("%s"%out)
				print("	takes %.2f seconds, total_adjustion is %d. original_cost:%f, after_adjust_cost:%f" % (time.time()-start_time, total_adjustion, original_total_cost, after_adjust_cost))
				print ("\n")
					
			else:
				while current_epoch == epoch.eval():
					pass

if __name__ == "__main__":
	tf.app.run()

