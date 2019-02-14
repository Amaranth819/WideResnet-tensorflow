#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 01:08:40 2019

@author: xulin
"""
import tensorflow as tf
import numpy as np
import cifar
import config

def continue_training(dataset_path, model_path, graph_path, ckpt_path, log_path, input_name, gt_name, epoch, bs, save_model_step):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    
    train_writer = tf.summary.FileWriter(log_path, sess.graph)
    merged = tf.summary.merge_all()
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name(input_name)
    gt = graph.get_tensor_by_name(gt_name)
    loss = graph.get_tensor_by_name("loss:0")
    global_steps = graph.get_tensor_by_name("global_step:0")
    learning_rate = graph.get_tensor_by_name("learning_rate:0")
    optimizer = graph.get_tensor_by_name("adam_minimizer:0")
    
    # dataset
    data = []
    label = []
    for cdir in dataset_path:
        dic = cifar.unpickle_cifar10(cdir)
        data.append(dic['data'])
        label.append(dic['labels'])
    data = np.concatenate(data, axis = 0)
    label = np.concatenate(label, axis = 0)
    print data.shape
    print label.shape
    dataset = cifar.create_dataset(data, label, bs)
    batch_tensor = cifar.get_next(dataset)
    
    e = 0
    while e < epoch:
        batch = sess.run(batch_tensor)
        if len(batch[1]) != bs:
            continue
        
        summary, gs, l, lr, _ = sess.run([merged, global_steps, loss, learning_rate, optimizer],
				feed_dict = {x : batch[0], gt : batch[1]})
        train_writer.add_summary(summary, gs)
        
        print "Global steps %d -- loss = %.6f, lr = %.9f" % (gs, l, lr)
        
        if gs % save_model_step == 0:
            saver.save(sess, model_path, global_step = save_model_step)
            print "Save the model successfully!"
            
        e += 1
            
    train_writer.close()
    
continue_training(
        dataset_path = config.dataset_dirs,
        model_path = config.model_dir,
        graph_path = config.model_dir + '-%d.meta' % config.save_model_step,
        ckpt_path = config.ckpt_dir,
        log_path = config.log_dir,
        input_name = "input:0",
        gt_name = "gt:0",
        epoch = 20000,
        bs = config.batch_size,
        save_model_step = config.save_model_step)