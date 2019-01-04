# Training Convolution Network for Classification

## Objective

The objective of this project is to build a simple machine learning model to implement a convolution network for
binary classification problem:

* Implement Convnet for classification in PyTorch.

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
1. Convolutional Neural Network is implemented and two more Fully-Connected layers is added at the
end of convolutional layer for classification purpose.
2. Since the problem is classification task so we use **Cross-Entropy** loss function.
3. Given training dataset is used but most importantly:
    * Data augmentation is performed such as scaling (double of input size to convnet) and randomly cropping the image to the input size
    * Also normalization is performed which is essential for swift and accelerated learning
    * **Note**: The implementation takes a liberty of getting all the training-data and respective labels from a CSV file and also for test images when performing prediction.


Others:

4. Optimizer such as SGD with momemtum, Adam e.t.c. for this task Adam is used.
5. For optimal learning rate paper by Leslie N. Smith
[A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
is used. This further allows expeditious training through its one-cycle policy and is really effective.


#### Testing/Prediciton
Trainined model is saved and which can be loaded whenever needed to perform prediction task.
The result of test data is saved in file <**test_output_prediction.csv**> which is self explanatory and <**train_output.log**> additionally contains classification report e.t.c.


**Note-1**: To perform training(<train.py>) and for testing/prediction (<predict.py>) as main module for each respective file.

#### TODO:
1. Extend for multiclass classification
2. More data augmentation
3. Use validation set for tuning hyperparameters of the model

**Note-2**: Upload images for running the model
