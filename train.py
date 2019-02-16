import sys
sys.path.append('./')

import tensorflow as tf
import numpy as np
import config
import wide_resnet as wrn
import cifar

class Train(object):
	def __init__(self, k, N, num_classes, bs, input_size, output_size, init_lr, lr_decay_step, lr_decay_rate, epoch, log_dir, dataset_dirs, model_dir, save_model_step):
		# input and output
		self.input = tf.placeholder(shape = input_size, dtype = tf.float32, name = "input")
		self.gt = tf.placeholder(shape = output_size, dtype = tf.int32, name = "gt")
		self.k = k
		self.N = N
		self.num_classes = num_classes
		self.bs = bs

		# learning rate
		self.init_lr = init_lr
		self.lr_decay_step = lr_decay_step
		self.lr_decay_rate = lr_decay_rate
		self.global_step = tf.Variable(0, trainable = False, name = "global_step")
		self.learning_rate = tf.train.exponential_decay(self.init_lr, self.global_step, self.lr_decay_step, self.lr_decay_rate, staircase = True, name = "learning_rate")
		self.save_model_step = save_model_step

		self.epoch = epoch
		self.log_dir = log_dir
		self.dataset_dirs = dataset_dirs
		self.model_dir = model_dir

	def loss_func(self, pred, gt):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = gt, logits = pred), name = "loss")
		tf.summary.scalar("loss", loss)
		return loss

	def evaluate(self, logits, labels, rank):
		correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")
		tf.summary.scalar("accuracy", accuracy)
		return accuracy

	def train(self):
		sess = tf.Session()

		# wrn.basic, wrn.bottle_neck, wrn.basic_wide, wrn.dropout
		pred = wrn.build_wide_resnet(self.input, self.num_classes, self.N, self.k, wrn.basic_wide, prob = 0.3)
		loss = self.loss_func(pred, self.gt)
		evaluation = self.evaluate(logits = pred, labels = self.gt, rank = 1)
		# optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name = "adam_optimizer").minimize(loss, global_step = self.global_step, name = "adam_minimizer")
		# weight decay is 0.0005
		optimizer_with_weight_decay = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.MomentumOptimizer)
		optimizer = optimizer_with_weight_decay(weight_decay = 0.0005, learning_rate = self.learning_rate, momentum = 0.9).minimize(loss, global_step = self.global_step, name = "momentum_minimizer")
		# optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9, name = "momentum_optimizer").minimize(loss, global_step = self.global_step, name = "momentum_minimizer")

		sess.run(tf.global_variables_initializer())

		train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
		merged = tf.summary.merge_all()
		saver = tf.train.Saver()

		# dataset
		data = []
		label = []
		for cdir in self.dataset_dirs:
			dic = cifar.unpickle_cifar10(cdir)
			data.append(dic['data'])
			label.append(dic['labels'])
		data = np.concatenate(data, axis = 0)
		label = np.concatenate(label, axis = 0)
		print data.shape
		print label.shape
		dataset = cifar.create_dataset(data, label, self.bs)
		batch_tensor = cifar.get_next(dataset)

		i = 0
		while i < self.epoch:
			batch = sess.run(batch_tensor)
			if len(batch[1]) != self.bs:
				continue

			summary, gs, l, eva, lr, _ = sess.run([merged, self.global_step, loss, evaluation, self.learning_rate, optimizer],
				feed_dict = {self.input : batch[0], self.gt : batch[1]})

			train_writer.add_summary(summary, gs)
			print "Global steps %d -- loss = %.6f, lr = %.9f, acc = %.6f" % (gs, l, lr, eva)

			if gs % self.save_model_step == 0:
				saver.save(sess, self.model_dir, global_step = self.save_model_step)
				print "Save the model successfully!"

			i += 1

		train_writer.close()


t = Train(config.k, config.N, config.num_classes, 
	config.batch_size, config.input_size, config.output_size, 
	config.init_lr, config.lr_decay_step, config.lr_decay_rate, 
	config.epoch, config.log_dir, config.dataset_dirs, config.model_dir, config.save_model_step)
t.train()
