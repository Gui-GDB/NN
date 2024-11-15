import numpy as np

from copy import deepcopy
from matplotlib import pyplot as plt
import argparse


def feed_forward(inputs, outputs, weights):
	"""
	前向传播的过程
	:param inputs: 输入数据
	:param outputs: 输出数据的正确值
	:param weights: 神经网络中的权重参数和偏置项
	:return: 损失值
	"""
	# 隐藏层1
	pre_hidden = np.dot(inputs, weights[0]) + weights[1]
	# Sigmod 非线性激活函数
	hidden = 1 / (1 + np.exp(-pre_hidden))
	# 输出层
	pred_output = np.dot(hidden, weights[2]) + weights[3]
	# 计算连续变量的损失值（loss）
	mean_squared_error = np.mean(np.square(pred_output - outputs))
	return mean_squared_error


def update_weights(inputs, outputs, weights, lr):
	"""
	反向传播更新权重参数
	:param inputs: 输入数据
	:param outputs: 输出的标准结果
	:param weights: 权重参数
	:param lr: 学习率（超参数）
	:return: 返回更新后的权重参数与当前权重参数的损失值
	"""
	# 定义每个权重参数增加的一个非常小的量(注意：这个不是 weight_decay)
	num = 0.005
	# 计算当前权重下的损失值
	original_loss = feed_forward(inputs, outputs, weights)
	# 创建一个原始权重参数的副本用来保存更新后的权重参数
	updated_weights = deepcopy(weights)
	# 遍历每一个参数进行更新
	for i, layer in enumerate(weights):
		for index, weight in np.ndenumerate(layer):
			temp_weights = deepcopy(weights)
			# 为神经网络中的每一个参数增加一个小的值
			temp_weights[i][index] += num
			# 计算更新一个权重后的损失值(每个参数都要去算一遍损失值，效率十分的低下，可以使用链式法则计算梯度下降)
			loss_plus = feed_forward(inputs, outputs, temp_weights)
			# 计算梯度
			grad = (loss_plus - original_loss) / num
			# 更新当前权重参数
			updated_weights[i][index] -= lr * grad
	return updated_weights, original_loss


def train(epochs, inputs, outputs, weights, lr):
	"""
	模型训练
	:param epochs: 迭代次数
	:param inputs: 输入数据
	:param outputs: 输出结果（标准答案）
	:param weights: 权重参数
	:param lr: 学习率
	:return: 损失值列表，最后的权重参数
	"""
	losses = []
	# 执行 100 个 epoch 内执行前向传播和反向传播。
	for epoch in range(epochs):
		weights, loss = update_weights(inputs, outputs, weights, lr)
		losses.append(loss)
	return losses, weights


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='手写简单的神经网络')
	parser.add_argument('--lr', '-l', type=float, default=0.005, help='learning rate')
	parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
	parser.add_argument('--seed', '-s', type=int, default=2003, help='random seed')
	args = parser.parse_args()

	# 输入数据
	inputs = np.array([[1, 1], [2, 1], [6, 4], [6, 9], [5, 3]])
	# 标准输出结果
	outputs = np.array([[2], [3], [10], [15], [8]])
	# 设置随机初始化种子
	np.random.seed(args.seed)
	# 随机初始化权重

	weights = [
		# 第一个参数数组对应于将输入层连接到隐藏层的 2x3 权重矩阵
		np.random.random((2, 3)),
		# 第二个参数数组表示与隐藏层的每个神经院相关的偏置值
		np.random.random((1, 3)),
		# 第三个参数数组对应于将隐藏层连接到输出层 3x1 权重矩阵
		np.random.random((3, 1)),
		# 最后一个参数数组表示与输出层相关的偏执值。
		np.random.random(1)
	]

	# 绘制损失值：
	losses, weights = train(args.epochs, inputs, outputs, weights, args.lr)
	plt.plot(losses)
	plt.title('Loss over increasing number of epochs')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()

	# 获取更新后的权重后，通过将输入传递给网络对输入进行预测并计算输出值。
	pred_data = [[1, 1], [3, 7], [2, 4]]
	print("需要预测的输入是：\n" + str(pred_data))
	pre_hidden = np.dot(np.array(pred_data), weights[0]) + weights[1]
	hidden = 1 / (1 + np.exp(-pre_hidden))
	pred_output = np.dot(hidden, weights[2]) + weights[3]
	print("预测的结果为：\n" + str(pred_output))
