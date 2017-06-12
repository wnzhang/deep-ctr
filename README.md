# Deep Learning for Ad CTR Estimation

NOTE: we have upgraded the code of this repository [here](https://github.com/Atomu2014/product-nets) with TensorFlow and more advanced models in our new paper [Product-based Neural Network for User Response Prediction](https://arxiv.org/abs/1611.00144).

This repository hosts the code of several proposed deep learning models for estimating ad click-through rates, implemented with `Theano`. The research paper [Deep Learning over Multi-field Categorical Data – A Case Study on User Response Prediction](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/deep-ctr.pdf) has been published on ECIR 2016.

Different from traditional deep learning tasks like image or speech recognition, where neural nets work well on continuous dense input features, for ad click-through rate estimation task, the input features are almost *categorical* and of *multiple field.* For example, the input context feature could be *City=London*, *Device=Mobile*. Such multi-field categorical features are always transformed into sparse binary features via one-hot encoding, normally millions of dimensions. Tranditional DNNs cannot work well on such input data beacuse of the large dimension and high sparsity.

This work tries to address the above problems and the experiment results are promising. The corresponding research paper "Deep Learning over Multi-Field Categorical Data: A Case Study on User Response Prediction" has been accepted and will be published in ECIR 2016.

Note that this is just the authors' *first* attempt of training DNN models to predict ad click-through rate. Significant efforts on research and engineering will be made further on this project.

More any questions please contact [Weinan Zhang](http://www0.cs.ucl.ac.uk/staff/w.zhang/) (w.zhang@cs.ucl.ac.uk) and Tianming Du (dutianming@quicloud.cn).

### Code Installation and Running

[Theano](http://deeplearning.net/software/theano/) and dependant packages (e.g., `numpy` and `sklearn`) should be pre-installed before running the code.

After package installation, you can simple run the code with the demo tiny dataset.
```
python FNN.py      # for FNN
python SNN_DAE.py  # for SNN_DAE
python SNN_RBM.py  # for SNN_RBM
```
The descriptions of the proposed models (FNN, SNN) are available in the above research paper, which will be available soon.

*Note:* directly running above code only checks the success of system installation. The input training/test are very small sample datasets, where the deep models are not effective. For large-scale datasets, please refer
* [iPinYou data formalizing repository](https://github.com/wnzhang/make-ipinyou-data)
* [Cretio 1T dataset](http://labs.criteo.com/downloads/download-terabyte-click-logs/)
* etc.

*Note:* In our further practice on very large data, the FM initialisation is not necessary any more to train a good FNN.
