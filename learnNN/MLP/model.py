import torch
import torch.nn as nn


class MLP(nn.Module):
	def __init__(self, input_dim, hid_dim1, hid_dim2, output_dim, dropout):
		super(MLP, self).__init__()

		# nn.Linear 是一个全连接层：y = Wx + b
		self.fc1 = nn.Linear(input_dim, hid_dim1)
		self.fc2 = nn.Linear(hid_dim1, hid_dim2)
		self.output = nn.Linear(hid_dim2, output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# 第一层：输入 x 经过 fc1 + ReLU 激活 + Dropout
		x = torch.relu(self.fc1(x))
		x = self.dropout(x)
		# 第二次：输入 x 经过 fc2 + ReLU 激活 + Dropout
		x = torch.relu(self.fc2(x))
		x = self.dropout(x)
		# 输出层：经过 output 层
		x = self.output(x)

		return x


