# WideResnet-tensorflow

Environment: python 2.7 and tensorflow 1.12 \
Dataset: Cifar10

2.16: Add weight decay and change the loss function to tf.train.MomentumOptimizer.

How to use:
1. Use the command "python train.py" to train a model. The training configuration can be found in config.py.
2. Look for the training model and log in the file directory ./model and ./log.
3. After a training, you can continue you training by using the command "python continue_training.py". In this file, you can set up your own epoch of continuing training.
