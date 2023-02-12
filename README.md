# Succesive Subspace Learning (SSL)
## _An efficient implemenation of SSL in PyTorch library_

This repo containes a very fast and effiecient implemenation of SSL using torch library. 

Please cite the following reference if you use `torch_SSL`:
```sh
@article{azizi2022pager,
  title={PAGER: Progressive Attribute-Guided Extendable Robust Image Generation},
  author={Azizi, Zohreh and Kuo, C-C Jay and others},
  journal={APSIPA Transactions on Signal and Information Processing},
  volume={11},
  number={1},
  year={2022},
  publisher={Now Publishers, Inc.}
}
```

## Usage
- Clone the repo and `cd` into it.
- Run `pip install torch`.
- Run `pip install numpy`.
- If you wish to force the code to run on CPU, open `torch_configs.py` and uncomment line #6. Otherwise, it will be run on GPU if available. 
- Run `torch_ssl.py` to see the example usage included under `if __name__=="__main__":`.
