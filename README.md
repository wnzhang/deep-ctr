# Deep Learning for Ad CTR Estimation
This repository is the authors' *first* attempt of training DNN models to predict ad click-through rate, implemented with `Theano`.

Different from traditional deep learning tasks like image or speech recognition, where neural nets work well on continuous dense input features, for ad click-through rate estimation task, the input features are almost *categorical* and of *multiple field.* For example, the input context feature could be *City=London*, *Device=Mobile*. Such multi-field categorical features are always transformed into sparse binary features via one-hot encoding, normally millions of dimensions. Tranditional DNNs cannot work well on such input data beacuse of the large dimension and high sparsity.

This work tries to address the above problems and the experiment results are promising. The corresponding research paper "Deep Learning over Multi-Field Categorical Data: A Case Study on User Response Prediction" has been accepted and will be published in ECIR 2016.

Note that this is just the authors' *first* attempt of training DNN models to predict ad click-through rate. Significant efforts on research and engineering will be made on this direction.

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
