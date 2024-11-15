import argparse

import torch
from utils import utils
import train
from model import MLP


def parse_args():
	# 创建 ArgumentParser 对象
	parser = argparse.ArgumentParser(description='Your script description goes here.')
	# 训练相关参数
	parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs ')
	parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
	# 设备配置
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training ')
	# 随机种子和训练控制
	parser.add_argument('--seed', type=int, default=2003, help='Random seed for reproducibility')
	# 模型保存路径
	parser.add_argument('--model_save_dir', type=str, default='./models', help='dir to save the model')
	# 解析命令行参数
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	# 解析命令行参数
	args = parse_args()
	# 设置cuda
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	print("using device: " + torch.cuda.get_device_name(0) if args.cuda else 'cpu')
	# 设置随机数种子，保证实验的可复现
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	# 获取数据加载器
	train_loader, val_loader = utils.get_data_MNIST(args.batch_size)
	# 初始化模型
	model = MLP(input_dim=28 * 28, hid_dim1=256, hid_dim2=128, output_dim=10, dropout=args.dropout)
	# 训练并评估模型
	trained_model = train.train_and_evaluate(model, train_loader, val_loader, args)
