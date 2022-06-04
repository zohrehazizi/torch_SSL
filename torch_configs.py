import torch
if torch.cuda.device_count()>0:
	device = 'cuda'
else:
	device = 'cpu'
# device = 'cpu'
def convert_to_torch(X):
	if type(X)!=torch.Tensor:
		return torch.tensor(X, dtype=torch.float64, device=device)
	else:
		return X.to(device)

def convert_to_numpy(x):
    if type(x)==torch.Tensor:
        return x.detach().cpu().numpy()
    else:
        return x