import tensorflow as tf
import numpy as np
import config
import cifar

# def evaluate(logits, labels, rank):
#     correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")
#     return accuracy

def test_error(graph_path, ckpt_path, test_data_path, input_name, gt_name, pred_name, bs):
    # prepare for testing dataset
    dic = cifar.unpickle_cifar10(test_data_path)
    dataset = cifar.create_dataset(dic['data'], dic['labels'], bs, 1)
    batch_tensor = cifar.get_next(dataset)
    
    # read pre-trained model
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(graph_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name(input_name)
        gt = graph.get_tensor_by_name(gt_name)
        pred = graph.get_tensor_by_name(pred_name)
        accuracy = graph.get_tensor_by_name("accuracy:0")
        a = []
        
        b = 1
        while True:
            try:
                batch = sess.run(batch_tensor)
                if len(batch[1]) != bs:
                    break
                acc = sess.run([accuracy], feed_dict = {x : batch[0], gt : batch[1]})
                a += acc
                print "The average accuracy of the %d batch is %.6f." % (b, np.mean(acc))
                b += 1
            except tf.errors.OutOfRangeError:
                break

        print "The average accuracy of all batches is %.6f." % np.mean(a)
            
test_error(graph_path = config.model_dir + "-%d.meta" % config.save_model_step,
           ckpt_path = config.ckpt_dir,
           test_data_path = config.test_dataset_dir,
           input_name = "input:0",
           gt_name = "gt:0",
           pred_name = "prediction:0",
           bs = config.batch_size)