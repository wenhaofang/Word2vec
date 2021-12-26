## Word To Vector Demo

This repository includes some demo word2vec models.

Note: The project refers to [动手学深度学习](https://zh.d2l.ai/)

Datasets:

* `dataset1`: text8
* `dataset2`: ptb

Models:

* `model1 (DONE)`: Continuous-Bag-Of-Words with Hierarchical-Softmax
* `model2 (DONE)`: Continuous-Bag-Of-Words with Negative-Sampling
* `model3 (DONE)`: Skip-Gram with Hierarchical-Softmax
* `model4 (DONE)`: Skip-Gram with Negative-Sampling
* `model5 (TODO)`: FastText
* `model6 (TODO)`: Glove

### Data Process

```shell
# download dataset text8
PYTHONPATH=. python dataprocess/process.py --dataset_name text8
# download dataset ptb
PYTHONPATH=. python dataprocess/process.py --dataset_name ptb
```

### Unit Test

* for loader

```shell
# CBOW_HS_Loader
PYTHONPATH=. python loaders/CBOW_HS_Loader.py
# CBOW_NS_Loader
PYTHONPATH=. python loaders/CBOW_NS_Loader.py
# SG_HS_Loader
PYTHONPATH=. python loaders/SG_HS_Loader.py
# SG_NS_Loader
PYTHONPATH=. python loaders/SG_NS_Loader.py
```

* for module

```shell
# CBOW_HS_Module
PYTHONPATH=. python modules/CBOW_HS_Module.py
# CBOW_NS_Module (the same as CBOW_HS_Module)
PYTHONPATH=. python modules/CBOW_NS_Module.py
# SG_HS_Module
PYTHONPATH=. python modules/SG_HS_Module.py
# SG_NS_Module (the same as SG_HS_Module)
PYTHONPATH=. python modules/SG_NS_Module.py
```

# Main Process

```shell
PYTHONPATH=. python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# CBOW_HS model
PYTHONPATH=. python main.py --module_type CBOW_HS --dataset_name text8
PYTHONPATH=. python main.py --module_type CBOW_HS --dataset_name ptb
```

```shell
# CBOW_NS model
PYTHONPATH=. python main.py --module_type CBOW_NS --dataset_name text8
PYTHONPATH=. python main.py --module_type CBOW_NS --dataset_name ptb
```

```shell
# SG_HS model
PYTHONPATH=. python main.py --module_type SG_HS --dataset_name text8
PYTHONPATH=. python main.py --module_type SG_HS --dataset_name ptb
```

```shell
# SG_NS model
PYTHONPATH=. python main.py --module_type SG_NS --dataset_name text8
PYTHONPATH=. python main.py --module_type SG_NS --dataset_name ptb
```
