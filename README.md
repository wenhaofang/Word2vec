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
