## [Effective Knowledge Transfer for Multi-Task Convolutional Architecture Search](http://none)

## Usage

#### Search on CIFAR-10 and CIFAR-100
The algorithm searchs on CIFAR-10 and CIFAR-100 by default:
```
cd cifar
python main.py 
```

#### Evaluation on CIFAR-10 and CIFAR-100
```
cd cifar/eval
python train.py --auxiliary --cutout --data ../dataset/CIFAR10 --set cifar10 --arch KT_C10 --init_channels 48 --gpu 0 # cifar-10
python train.py --auxiliary --cutout --data ../dataset/CIFAR100 --set cifar100 --arch TREMT_C100 --init_channels 42 --gpu 0 # cifar-100
```

#### Search on MNIST and Fashion-MNIST
```
cd mnist
python main4mnist.py 
```

#### Test on MNIST and Fashion-MNIST
```
cd mnist
python test4mnist.py # need to replace manually the parameter '--log_dir'
```

#### Search on MedMNIST
```
cd med
python main4med.py 
```

#### Test on MedMNIST
```
cd med
python test4med.py # need to replace manually the parameter '--log_dir'
```

## File structure of this example
+ Path of the datasets:
  - `./dataset/`
+ Experiments:
  - `./cifar/`
  - `./mnist/`
  - `./med/`
+ Search space (cifar example):
  - `./cifar/models/`
  - `./cifar/modules/`
  - `./cifar/cnn_utils.py`
  - `./cifar/common.py`
  - `./cifar/data.py`
+ Algorithm related:
  - `./cifar/gnas/`
  - `./cifar/graph/`
  - `./cifar/transfer_rank.py`
+ Parameter settings:
  - `./cifar/config.py`
+ Search results:
  - `./cifar/logs/`
+ Evaluation:
  - `./cifar/eval/`
+ Master file:
  - `./cifar/main.py`
+ Paper related:
  - `./figures/`
 
## Requirements
| Package   | Version  |  Note|
| :------------- | :----------: | :----------: | 
| python |   3.8.0   | 
| torch |   1.9.0+cu111   | 
| torchvision  |    0.10.0+cu111     |
| networkx | 2.8.8 |
| gensim | 4.3.0 |  
| medmnist |   2.2.3   | 
| graphviz   | 2.40.1 | Drawing the structure of the cells |
| pygraphviz | 1.7 | Drawing the structure of the cells |



## Related work

[EMT-NAS:Transferring Architectural Knowledge Between Tasks From Different Datasets](https://github.com/PengLiao12/EMT-NAS)

[node2vec: Scalable Feature Learning for Networks](https://github.com/aditya-grover/node2vec)


## Reference
If you find this code helpful for your research, please cite our paper.
```
none
```
