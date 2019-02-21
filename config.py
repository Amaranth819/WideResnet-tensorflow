# data parameters
batch_size = 128
image_size = [32, 32, 3]
ori_size = [3, 32, 32]
num_classes = 10
input_size = [batch_size] + image_size
output_size = [batch_size, num_classes]
k = 2
N = 2

# training
init_lr = 0.1
lr_decay_rate = 0.2
lr_decay_step = 10000

epoch = 20000
save_model_step = 1000
log_dir = "./log/"
model_dir = "./model/wideresnet"
ckpt_dir = "./model/"
dataset_dirs = ["./cifar/cifar-10-batches-py/data_batch_1",
	"./cifar/cifar-10-batches-py/data_batch_2",
	"./cifar/cifar-10-batches-py/data_batch_3",
	"./cifar/cifar-10-batches-py/data_batch_4",
	"./cifar/cifar-10-batches-py/data_batch_5"]
test_dataset_dir = "./cifar/cifar-10-batches-py/test_batch"
