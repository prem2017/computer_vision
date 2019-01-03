# Training Simple Fully Convolution Network for Segmentation

## Objective

Machine Learning model to implement simple convolution network for
binary segmentation problem:

* Implement Convnet for segmentation in PyTorch.

* Implementation should allow the model to be trained on GPU i.e. to make it device agnostic.


## Implementation

### Dependency:
1. PyTorch >= 0.4
2. Python >= 3.6
3. Other library such as **skimage**, **sklearn**, **numpy**, **pandas** e.t.c.


### Operations
To implement any deep neural network in pytorch three fundamental module is needed i.e. a Neural Network modelling function,
a function to measure the loss, training dataset, and
others factors such as optimizer for accelerated and regularized learning, optimum selection of learning rate,
dropouts and weight-decay for generalization of model.
#### Training
1. Neural Network is implemented as per objective which is fully convolutional layer for segmentation purpose.
2. Since the problem is segmentation classification task so we use **Cross-Entropy** loss function.
3. This is demo code only suitable for binary class segmentation. Nevertheless, data augmentation option is also provided for robust training (resizing is only validated in case of binary class segmentation only)
Moreover, one can check the crude performance of model as well.
4. Also normalization (standardization) is performed which is essential for swift and accelerated learning. It should be noted that each image is
standerized according to its own pixel content thus not requiring to store any mean/variance set.

Others:

5. Optimizer such as SGD with momemtum, Adam e.t.c. for this task Adam is used.
6. For optimal learning rate paper by Leslie N. Smith
[A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
is used. This further allows expeditious training through its one-cycle policy and is really effective.


#### Testing/Prediciton
Trained model is saved which can be loaded whenever needed to perform prediction task.
The result of test data is saved in directory <**result_dir**>.


**Note**:
1. To perform training and testing use <**train.py**> and <**predict.py**> respectively for corresponding task.
2. Works only for binary class segmentation

**TODO:**
1. Extend for multiclass segmentation
2. Also extendent for [instance segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo&t=2s) similar to object detection
i.e. hybrid between semantic segmentation and object detection.





