# Matrix Factorization for Collaborative Filtering with Side Features

This is my PyTorch implementation of a Matrix Factorization model for Collaborative Filtering. This model adds in "side" features, especially useful in cold-start situations.

## Background
A common challenge in collaborative filtering is the cold start problem due to its inability to address new items and new users. Or many users are supplying very few ratings, making the user-item interaction matrix very sparse. A way to relieve this problem is to incorporate additional sources of information about the users, aka side features. These can be user attributes (demographics) and implicit feedback.  

let’s say I know the occupation of the user. I have two choices for this side feature: adding it as a bias (artists like movies more than other occupations) and adding it as a vector (realtors love real estate shows). The matrix factorization model should integrate all signal sources with enhanced user representation, as seen in equation 5:
$$R_{ui} = q_i*p_u+q_i*t_o + b + w_i + w_u +d_o $$ 

where:
- the bias for occupation is denoted by $d_o$ meaning that occupation changes like the rating (not implemented in the code)
- the occupation vector is denoted by $t_o$ - having a weight per item for the occupation impact

and the loss function: 
$$\sum{(r_{ui}-q_i*p_u-q_i*t_o-b-w_i-w_u-d_o)^2+Regularize(q_i+p_u+t+o+w_i+w_o)} $$





## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Scripts
* [loader.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features/loader.py): This is the script that loads the data.
* [MFSideFeat.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features/MFSideFeat.py): This is the model script that defines the Matrix Factorization model with side features.
* [train.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/MetaRec/tree/master/Matrix-Factorization-Experiments/MF-Side-Features/results).
After training the model for 50 epochs with 75/25 train-test split, I got the training loss MSE = 0.6602 and test loss MSE = 0.7843 with training time = 13m34s.

## Run Tensorboard In The Background
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models.
It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

Here is the Mean Squared Error Loss on the training set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features/loss_mse.svg" width="1000" />

Here is the Mean Squared Error Loss on the test set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features/validation_avg_loss.svg" width="1000" />
