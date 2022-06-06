### Written by: Zohreh Azizi ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_configs import device, convert_to_torch, convert_to_numpy
#device = 'cpu'

def remove_mean(features, dim):
    feature_mean = torch.mean(features, dim=dim, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean

class PCA(nn.Module):
	def __init__(self, n_components, svd_solver = 'full'):
		super().__init__()
		self.n_components = n_components
		self.svd_solver = svd_solver
		self.kernels = nn.parameter.Parameter(None, requires_grad=False)
		self.mean = nn.parameter.Parameter(None, requires_grad=False)
	
	def fit(self,X):
		X = convert_to_torch(X)
		X, self.mean.data = remove_mean(X, dim=0)
		U,s,V = torch.linalg.svd(X, full_matrices=False)
		if self.n_components>1:
			rank = int(self.n_components)
		else:
			rank = None
			for i in range(len(s)):
				if torch.sum(s[0:i])/torch.sum(s)>=self.n_components:
					rank = i
					break
		if rank is None:
			rank = V.size(0)-1
		self.kernels.data = V[:rank+1]
		
	def fit_transform(self, X):
		X = convert_to_torch(X)
		X, self.mean.data = remove_mean(X, dim=0)
		U,s,V = torch.linalg.svd(X, full_matrices=False)
		if self.n_components>1:
			rank = int(self.n_components)
		else:
			rank = None
			for i in range(len(s)):
				if torch.sum(s[0:i])/torch.sum(s)>=self.n_components:
					rank = i
					break
		if rank is None:
			rank = V.size(0)-1
		self.kernels.data = V[:rank+1]
		return convert_to_numpy(torch.matmul(X, self.kernels.data.transpose(1,0)))
	
	def transform(self,X):
		X = convert_to_torch(X)
		X = X-self.mean 
		return convert_to_numpy(torch.matmul(X, self.kernels.data.transpose(1,0)))
	
	def inverse_transform(self,X):
		X = convert_to_torch(X, self.mean.device)
		return convert_to_numpy(torch.matmul(X,self.kernels.data)+self.mean)

class torchSaab(nn.Module):
	def __init__(self, kernel_size, stride, channelwise):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride 
		self.feature_expectation = torch.nn.parameter.Parameter(data=None, requires_grad=False)
		self.ac_kernels = torch.nn.parameter.Parameter(data=None, requires_grad=False)
		self.compute_mode = 'saab'
		self.bias = torch.nn.parameter.Parameter(data=None, requires_grad=False)
		self.channelwise = channelwise
		self.in_w = None 
		self.in_h = None 
		self.in_channels = None 
		self.w = None 
		self.h = None 
		self.out_channels = None

	def set_param_sizes(self, input_size):
		self.in_channels, self.in_w, self.in_h = input_size
		self.w = (self.in_w-self.kernel_size[0])//self.stride[0]
		self.h = (self.in_h-self.kernel_size[1])//self.stride[1]
		self.out_channels = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
		
	
	def fit(self, x, bias=None):
		if self.channelwise:
			return self.fit_channelwise(x, bias)
		x = convert_to_torch(x)

		self.in_w = x.size(2) 
		self.in_h = x.size(3)
		self.in_channels = x.size(1)
		self.w = (self.in_w-self.kernel_size[0])//self.stride[0]+1
		self.h = (self.in_h-self.kernel_size[1])//self.stride[1]+1
		sample_patches = F.unfold(x, self.kernel_size, stride=self.stride) #<n, dim, windows>
		n, dim, windows = sample_patches.size()
		sample_patches = sample_patches.permute(0,2,1)#<n, windows, dim>
		sample_patches = sample_patches.reshape(n*windows, dim)#<n.windows, dim>
		rank_ac = dim-1
		sample_patches_ac, dc = remove_mean(sample_patches, dim=1)  # Remove patch mean
		training_data, feature_expectation = remove_mean(sample_patches_ac, dim=0)
		self.feature_expectation.data = feature_expectation
		U,s,V = torch.linalg.svd(training_data, full_matrices=False)
		self.ac_kernels.data = V[:rank_ac]
		num_channels = self.ac_kernels.size(-1)
		dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
		ac = torch.matmul(training_data, torch.transpose(self.ac_kernels, 1,0))
		transformed = torch.cat([dc, ac], dim=1)
		bias = torch.max(torch.norm(transformed, dim=1))
		feature_layer = transformed.reshape(n, self.w, self.h, -1) # -> [num, w, h, c]
		feature_layer = feature_layer.permute(0,3,1,2) # -> [num, c, w, h]
		self.out_channels = feature_layer.size(1)
		return feature_layer, bias

	def fit_channelwise(self, x, bias):
		x = convert_to_torch(x)
		if bias.ndim!=0:
			bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
		self.bias.data = bias
		x = x+bias #<n,c,w,h>
		n, self.in_channels, self.in_w, self.in_h = x.size()
		self.w = (self.in_w-self.kernel_size[0])//self.stride[0]+1
		self.h = (self.in_h-self.kernel_size[1])//self.stride[1]+1
		sample_patches = F.unfold(x, self.kernel_size, stride=self.stride) #<n, dim, windows>, where dim=c.kernelsize[0].kernelsize[1]
		sample_patches = sample_patches.permute(0,2,1)#<n, windows, dim>
		sample_patches = sample_patches.reshape(n, self.w, self.h, self.in_channels, -1) #<n, w, h, c, k0k1>dim = self.kernel_size[0]*self.kernel_size[1]
		sample_patches = sample_patches.permute(3,0,1,2,4) #<c, n, w, h, k.k1>
		sample_patches = sample_patches.reshape(self.in_channels, -1, self.kernel_size[0]*self.kernel_size[1])#<c, nwh, k0k1>
		sample_patches_ac, dc = remove_mean(sample_patches, dim=2)
		training_data, feature_expectation = remove_mean(sample_patches_ac, dim=1)
		self.feature_expectation.data = feature_expectation
		rank_ac = self.kernel_size[0]*self.kernel_size[1]-1
		U,s,V = torch.linalg.svd(training_data, full_matrices=False)
		self.ac_kernels.data = V[:,:rank_ac,:]
		num_channels = self.ac_kernels.shape[-1]
		dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
		ac = torch.matmul(training_data, self.ac_kernels.permute(0,2,1))
		transformed = torch.cat([dc, ac], dim=2) #<c, n.windows, k0.k1>
		next_bias =torch.norm(transformed, dim=2)
		next_bias,_ = torch.max(next_bias,dim=1)
		transformed = transformed.reshape(self.in_channels,n,self.w,self.h,self.kernel_size[0]*self.kernel_size[1])
		transformed = transformed.permute(1,0,4,2,3)
		transformed = transformed.reshape(n,-1, self.w,self.h)
		next_bias = next_bias.repeat(self.kernel_size[0]*self.kernel_size[1],1)
		next_bias = next_bias.transpose(1,0)
		next_bias = next_bias.ravel()
		self.out_channels = transformed.size(1)
		return transformed, next_bias

	def forward(self, x):
		if self.compute_mode=='saab':
			if self.channelwise:
				return self.saab_channelwise(x)
			else:
				return self.saab(x)
		elif self.compute_mode=='inverse_saab':
			if self.channelwise:
				return self.inverse_saab_channelwise(x)
			else:
				return self.inverse_saab(x)
		else:
			assert False
	
	def saab(self, x):
		if self.channelwise:
			return self.saab_channelwise(x)
		x = convert_to_torch(x)
		assert x.size(1)==self.in_channels
		assert x.size(2)==self.in_w
		assert x.size(3)==self.in_h
		sample_patches = F.unfold(x, self.kernel_size, stride=self.stride) #<n, dim, windows>
		n, dim, windows = sample_patches.size()
		sample_patches = sample_patches.permute(0,2,1)#<n, windows, dim>
		sample_patches = sample_patches.reshape(n*windows, dim)#<n.windows, dim>
		sample_patches_ac, dc = remove_mean(sample_patches, dim=1)  # Remove patch mean
		sample_patches_centered_ac = sample_patches_ac - self.feature_expectation
		num_channels = self.ac_kernels.size(-1)
		dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
		ac = torch.matmul(sample_patches_centered_ac, self.ac_kernels.transpose(1,0))
		transformed = torch.cat([dc, ac], dim=1)
		feature_layer = transformed.reshape(n, self.w, self.h, -1) # -> [num, w, h, c]
		feature_layer = feature_layer.permute(0,3,1,2) # -> [num, c, w, h]
		return feature_layer
	
	def saab_channelwise(self, x):
		x = convert_to_torch(x)
		assert x.size(1)==self.in_channels
		assert x.size(2)==self.in_w
		assert x.size(3)==self.in_h
		x = x+self.bias #<n,c,w,h>
		n, self.in_channels, self.in_w, self.in_h = x.size()
		self.w = (self.in_w-self.kernel_size[0])//self.stride[0]+1
		self.h = (self.in_h-self.kernel_size[1])//self.stride[1]+1
		sample_patches = F.unfold(x, self.kernel_size, stride=self.stride) #<n, dim, windows>, where dim=c.kernelsize[0].kernelsize[1]
		sample_patches = sample_patches.permute(0,2,1)#<n, windows, dim>
		sample_patches = sample_patches.reshape(n, self.w, self.h, self.in_channels, -1) #<n, w, h, c, k0k1>dim = self.kernel_size[0]*self.kernel_size[1]
		sample_patches = sample_patches.permute(3,0,1,2,4) #<c, n, w, h, k.k1>
		sample_patches = sample_patches.reshape(self.in_channels, -1, self.kernel_size[0]*self.kernel_size[1])#<c, nwh, k0k1>
		sample_patches_ac, dc = remove_mean(sample_patches, dim=2)
		sample_patches_ac = sample_patches_ac-self.feature_expectation
		num_channels = self.ac_kernels.shape[-1]
		dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
		ac = torch.matmul(sample_patches_ac, self.ac_kernels.permute(0,2,1))
		transformed = torch.cat([dc, ac], dim=2) #<c, n.windows, k0.k1>
		transformed = transformed.reshape(self.in_channels,n,self.w,self.h,self.kernel_size[0]*self.kernel_size[1])
		transformed = transformed.permute(1,0,4,2,3)
		transformed = transformed.reshape(n,-1, self.w,self.h)
		return transformed

	def inverse_saab(self,x):
		if self.channelwise:
			return self.inverse_saab_channelwise(x)
		x = convert_to_torch(x)
		assert x.size(2)==self.w 
		assert x.size(3)==self.h
		sample_images =  x.permute(0,2,3,1) #<n, w, h, c>
		n,w,h,c=sample_images.size()
		sample_patches = sample_images.reshape(n*w*h,c)
		dc_comp = sample_patches[:, 0:1] * 1.0 / np.sqrt(self.ac_kernels.shape[-1])
		ac_comp = torch.matmul(sample_patches[:, 1:], self.ac_kernels)
		sample_rec = dc_comp + (ac_comp + self.feature_expectation)
		sample_rec = sample_rec.reshape(n, -1, sample_rec.size(1))
		sample_rec = sample_rec.permute(0,2,1)
		out = F.fold(sample_rec, output_size=(self.in_w, self.in_h), kernel_size=self.kernel_size, stride=self.stride)
		return out
	
	def inverse_saab_channelwise(self,x):
		x = convert_to_torch(x)
		assert x.size(2)==self.w 
		assert x.size(3)==self.h
		n = x.size(0)
		x = x.reshape(x.size(0), self.in_channels, -1, self.w, self.h)
		x = x.permute(1,0,3,4,2)
		x = x.reshape(self.in_channels, -1, x.size(-1))
		dc_comp = x[:,:, 0:1]* 1.0 / np.sqrt(self.ac_kernels.size(-1))
		ac_comp = torch.matmul(x[:, :, 1:], self.ac_kernels)
		sample_rec = dc_comp + (ac_comp + self.feature_expectation)
		sample_rec = sample_rec.reshape(self.in_channels, n, -1, sample_rec.size(2))
		sample_rec = sample_rec.permute(1,2,0,3)###??????????????????
		sample_rec = sample_rec.reshape(sample_rec.size(0), sample_rec.size(1), -1)
		sample_rec = sample_rec.permute(0,2,1)
		out = F.fold(sample_rec, output_size=(self.in_w, self.in_h), kernel_size=self.kernel_size, stride=self.stride)
		return out-self.bias
	
	def extra_repr(self):
		return f"input shape: ({self.in_channels}, {self.in_w}, {self.in_h})  kernel size: {self.kernel_size}  stride: {self.stride} outut shape: ({self.out_channels}, {self.w}, {self.h}) channelwise: {self.channelwise}"
	
class sslModel(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.layers = nn.ModuleList(layers)
		self.compute_mode = "saab"
	def forward(self, x):
		x = convert_to_torch(x)
		if self.compute_mode=="saab":
			for l in self.layers:
				x = l.saab(x)
		elif self.compute_mode=='inverse_saab':
			for l in self.layers[::-1]:
				x = l.inverse_saab(x)
		else:
			assert False
		return convert_to_numpy(x)
	def fit(self, x):
		x = convert_to_torch(x)
		bias = None
		for i, l in enumerate(self.layers):
			l.compute_mode = "saab"
			x, bias = l.fit(x,bias)
			print(f"learned parameters for layer {i}, output size was {x.size()}")
		return convert_to_numpy(x)
	def set_forward(self):
		self.compute_mode = "saab"
		for l in self.layers:
			l.compute_mode = "saab"
	def set_inverse(self):
		self.compute_mode = "inverse_saab"
		for l in self.layers:
			l.compute_mode = "inverse_saab"
	def get_size(self):
		num_params = 0 
		for p in self.parameters():
			num_params+=torch.prod(torch.tensor(p.size())).type(torch.DoubleTensor).item()
		return num_params


if __name__=="__main__":
	from generation_modules import timerClass

	x = np.random.rand(1000,1,32,32)
	timer = timerClass()
	layer1 = torchSaab(kernel_size=(4,4), stride=(4,4), channelwise=False)
	layer2 = torchSaab(kernel_size=(2,2), stride=(2,2), channelwise=True)
	layer3 = torchSaab(kernel_size=(2,2), stride=(2,2), channelwise=True)
	layer4 = torchSaab(kernel_size=(2,2), stride=(2,2), channelwise=True)
	model = sslModel([layer1, layer2, layer3, layer4])
	model.fit(x)
	timer.register("torch_model fit completed")
	print(model)
	features = model(x)
	timer.register("torch model forward computed")
	print(f"features size was {features.shape}")
	model.set_inverse()
	x_reconst = model(features)
	diff = np.mean((np.abs(x_reconst-x)/(x+1e-10)))
	print(f"mean reconstruction error was {diff}")
	print(f"model size was {model.get_size()}")
	timer.register("torch model backward computed")
	timer.print()

	print("###########################")
