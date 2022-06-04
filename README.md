# Succesive Subspace Learning (SSL)
## _An efficient implemenation of SSL using PyTorch library_

This repo containes a very fast and effiecient implemenation of SSL using torch library. 

Please cite the following reference if you use `torch_SSL`:
```sh
https://arxiv.org/abs/2206.00162
```

## Usage
- Clone the repo and `cd` into it.
- Run `pip install torch`.
- Run `pip install numpy`.
- If you wish to force the code to run on CPU, open `torch_configs.py` and uncomment line #6. Otherwise, if will be run on GPU if available. 
- Run `torch_ssl.py` to see the example usage included under `if __name__=="__main__":`.
