# data parameters
batch_size = 128
image_size = [32, 32, 3]
num_classes = 10
input_size = [batch_size] + image_size
output_size = [batch_size, num_classes]
k = 10
N = 4

# training
init_lr = 0.1
lr_decay_rate = 0.8
lr_decay_step = 60

epoch = 200
save_model_step = 200
log_dir = "./log/"
model_dir = "./model/wideresnet"
ckpt_dir = "./model/"
dataset_dirs = ["../../../dataset/cifar/cifar-10-batches-py/data_batch_1",
	"../../../dataset/cifar/cifar-10-batches-py/data_batch_2",
	"../../../dataset/cifar/cifar-10-batches-py/data_batch_3",
	"../../../dataset/cifar/cifar-10-batches-py/data_batch_4",
	"../../../dataset/cifar/cifar-10-batches-py/data_batch_5"]
test_dataset_dir = "../../../dataset/cifar/cifar-10-batches-py/test_batch"