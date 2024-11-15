import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse


class MyNeuraNet(torch.nn.Module):
	"""
	创建神经网络
	继承自 nn.Module, nn.Module 是所有神经网络模块的基类
	"""
	def __init__(self, inputs, outputs, hidden):
		"""
		使用 __init__ 方法初始化神经网络的所有组件
		调用 super().__init__() 确保类继承 nn.Module，可以利用 nn.Module 编写的所有预构建函数。
		"""
		super().__init__()
		# 全连接层(包含了偏置参数)
		self.input_to_hidden_layer = nn.Linear(in_features=inputs, out_features=hidden)
		# 使用 ReLU 激活函数
		# self.hidden_layer_activation = nn.ReLU()
		# 使用 Sigmod 激活函数
		self.hidden_layer_activation = nn.Sigmoid()
		# 全连接层(包含了偏置参数)
		self.hidden_to_output_layer = nn.Linear(in_features=hidden, out_features=outputs)

	def forward(self, x):
		"""
		将初始化后的神经网络组件连接在一起，并定义网络的前向传播方法 forward
		必须使用 forward 作为前向传播的函数名，因为 PyTorch 保留此函数作为执行前向传播的方法，使用其他名称会引发错误。
		:param x: 输入数据
		:return: 返回前向传播预测的结果
		"""
		# 等价于 x @ self.input_to_hidden_layer
		x = self.input_to_hidden_layer(x)
		x = self.hidden_layer_activation(x)
		x = self.hidden_to_output_layer(x)
		return x


class MyDataSet(Dataset):
	"""
	在 MyDataSet类中，存储数据信息以便可以将一批（batch）数据点捆绑在一起（使用 DataLoader），并通过一次前向和反向传播更新权重。
	"""
	def __init__(self, x, y):
		"""
		该方法接受输入和输出，并将它们转换为 torch 浮点对象
		:param x: 输入数据
		:param y: 输出数据
		"""
		# torch.tensor(x).float()
		self.x = x.clone().detach()
		self.y = y.clone().detach()

	def __getitem__(self, index):
		"""
		获取数据样本，
		:param index: 为从数据集中获取到的索引
		:return: 返回获取到数据样本
		"""
		return self.x[index], self.y[index]

	def __len__(self):
		"""
		指定数据长度
		:return: 返回输入数据的长度
		"""
		return len(self.x)


def train(epoch, lr, dataLoader):
	# 定义损失函数，由于需要预测连续变量，因此使用均方误差作为损失函数：
	loss_fn = nn.MSELoss()
	# 定义用于降低损失值的优化器，优化器的输入是与神经网络相对应的参数（权重与偏置）以及更新权重时的学习率。
	optimizer = Adam(myNet.parameters(), lr=lr)
	# 保存每次迭代的损失值
	loss_values = []
	for _ in range(1, epoch + 1):
		for batch in dataLoader:
			x, y = batch
			# 梯度清零
			optimizer.zero_grad()
			# 计算损失值
			loss_value = loss_fn(myNet(x), y)
			# 梯度下降
			loss_value.backward()
			# 更新权重
			optimizer.step()
			loss_values.append(loss_value.item())
	return loss_values


if __name__ == '__main__':
	parser = argparse.ArgumentParser("PyTorch 构建简单的神经网络")
	parser.add_argument('-l', '--lr', type=float, default=0.005, help='learning rate')
	parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size')
	parser.add_argument('-s', '--seed', type=int, default=2003, help='random seed')
	args = parser.parse_args()

	# 初始化数据集，定义输入（x）和输出（y）,输入中的每个列表的值之和就是输出列表中对应的值
	x = np.array([[1, 1], [2, 1], [6, 4], [6, 9], [5, 3]])
	# 标准输出结果
	y = np.array([[2], [3], [10], [15], [8]])
	# 将输入列表对象转换为张量对象
	x = torch.tensor(x, dtype=torch.float)
	y = torch.tensor(y, dtype=torch.float)
	# 获取当前是否有可以使用的 cuda 设备
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# 将输入数据和输出数据点注册到 device 中。
	x = x.to(device)
	y = y.to(device)

	# 指定随机种子，保证每次神经网络都使用的是相同的随机值，利于代码复现
	torch.manual_seed(args.seed)
	if device == 'cuda':
		torch.cuda.manual_seed(args.seed)
		print('using device: ' + torch.cuda.get_device_name(0))

	# 创建 MyNeuralNet 类对象的实例并将其注册到 device
	myNet = MyNeuraNet(2, 1, 3).to(device)

	# 实例化数据集
	myDataset = MyDataSet(x, y)
	# 通过 DataLoader 传递数据实例，从原始输入输出对象中获取 batch_size 个数据点
	dataLoader = DataLoader(dataset=myDataset, batch_size=args.batch_size, shuffle=True)

	# 绘制损失函数
	loss_values = train(args.epochs, args.lr, dataLoader)
	plt.plot(loss_values)
	plt.title("Loss variation over increasing epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Loss value")
	plt.show()

	# 利用训练的模型预测
	pred_data = [[1, 1], [3, 7], [2, 4]]
	print("需要预测的输入是：\n" + str(pred_data))
	pred_data = torch.tensor(pred_data, dtype=torch.float).to(device)
	print("预测的结果为：\n" + str(myNet(pred_data)))
