# Matrix Factorization for Collaborative Filtering with Temporal Features

This is my PyTorch implementation of a Matrix Factorization model for Collaborative Filtering. This model includes temporal effects which can track seasonal and periodic changes.

## Background

So far, our matrix factorization models have been static. In reality, item popularity and user preferences change constantly. Therefore, we should account for the temporal effects reflecting the dynamic nature of user-item interactions. To accomplish this, we can add a temporal term that affects user preferences and, therefore, the interaction between users and items.

To mix it up a bit, let’s try out a new equation 7 below with dynamic prediction rule for a rating at time t:  
$$ R_{ui}(t)=q_i*p_u+p_u*t_o+p_u(t)+b+w_i+w_u$$

where: 
- $p_u(t)$ takes user factors as a function of time. on the other hand, $q_i$ stays the same because items are static
- we include occupation changes depending on the user ($p_u*t_o$)


so the loss function:
$$\sum{(R_{ui}-q_i*p_u-p_u*t_o-b-w_i-w_u-d_o)^2+Regularize(q_i+p_u+t+w_i+w_o+p_u(t))} $$

<font color='pink'> - what is $d_o$ ? where is $p_u(t)$? </font>
<font color='yellow'> $d_o$ is the occupation bias (we dont see it in the code) </font>

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
* [loader.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Temporal-Features/loader.py): This is the script that loads the data.
* [MFTemporalFeat.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Temporal-Features/MFTemporalFeat.py): This is the model script that defines the Matrix Factorization model with temporal features.
* [train.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Temporal-Features/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/MetaRec/tree/master/Matrix-Factorization-Experiments/MF-Temporal-Features/results).
After training the model for 50 epochs with 75/25 train-test split, I got the training loss MSE = 0.7088 and test loss MSE = 0.7939 with training time = 18m51s.

## Run Tensorboard In The Background
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models.
It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

Here is the Mean Squared Error Loss on the training set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Temporal-Features/loss_mse.svg" width="1000" />

Here is the Mean Squared Error Loss on the test set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Temporal-Features/validation_avg_loss.svg" width="1000" />
