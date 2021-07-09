# Rain in Australia
### Predicting next-day rain in Australia using an LSTM Neural Network

A dataset from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) <br>

### Best achieved result: F1 score of 0.78

<br>

## Contents

Project is split into two key pieces - data preparation and modelling.

**Data preparation notebook**
- does some data exploration,
- cleans up several variables,
- creates some new ones,
- imputes missing data,
- splits up data for train/test,
- scales the data

Extra care is taken to avoid any data leakage. Notebook runs once and saves the outputs as csvs.

**Data modelling notebook**
- reads in the cleaned csvs
- transforms the data into an appropriate shape for the model to read in
- throws it into a dataloader for batched processing
- sends the data through a training loop
- a number of custom functions and the LSTM model itself is stored in the `custom.py` script
- all of the metadata + artefacts (such as loss/accuracy curves) are stored in [wandb](https://wandb.ai/vinas/RainAustralia):

<img src="./Plots/wandb.png">

<br>

## Report

### Motivation  
<div align="justify">
This data is of a time series nature. It is reasonable to assume that the weather inhibits patterns over
several days therefore modelling this data with a model built for sequential data could work. A single
layer LSTM Neural Network is chosen as it is widely used and is often successful.
</div>

### Data transformation & LSTM settings  

<div align="justify">
The data needs to be reshaped to fit the LSTM input requirements. First, the dataset is split by city.
Then, arrays of lenght 7 are created, containing the 7-day history of all 22 features prior to the target.
These arrays are then stacked into a tensor of shape [N, 7, 22] where N represents the total number of
observations. This is done for both training and test data.
The chosen LSTM architecture is simple - a single layer with 5 hidden nodes. The output then goes
through a ReLu activation function and finally into a dense linear layer with one output neuron. A
sigmoid function is then applied in order to get the prediction. Binary Cross Entropy is used as the loss
function for training. The network is trained for 500 epochs in batches of 1024 while keeping track of the
loss values and the performance metrics.
</div>

### Results  
<div align="justify">
There were 5 different model runs in total. Each run was a slight modification of the previous one in effort
to improve performance. Two of these runs (lively-voice-2 - the baseline and autumn-gorge-4 - the best run) are described below.
</div>

#### Base run

<figure>
  <img src="./Plots/run2.png" alt="fig1"/>
  <figcaption>Figure 1: Base run using all data (lively-voice-2 from wandb)</figcaption>
</figure> <br>
<br>

<div align="justify">
The default network trained for the whole of 500 epochs though improvements were minimal past epoch 100.
While loss continued to decrease, the change in accuracy was negligible reaching 0.86 for test and 0.87
for training. Similarly, F1 score for test data kept oscillating past epoch 100 reaching an average of
0.77. However, while overall accuracy is reasonably high, the confusion matrix in Fig 1 indicates stark
differences in prediction performance for the majority/minority groups. Furthermore the classification
report in Table 1 shows that while the F1 score for the majority class was 0.91 it reached only 0.63 for
the minority group. Similarly, the recall rate for the minority class was very low at just 0.55 indicating
sub-optimal model performance.
</div>

                    precision    recall  f1-score   support  

               0.0       0.88      0.95      0.91     31774  
               1.0       0.74      0.55      0.63      8925  

          accuracy                           0.86     40699  
         macro avg       0.81      0.75      0.77     40699  
      weighted avg       0.85      0.86      0.85     40699
                
    Table 1: Classification report for the base run

<div align="justify">
The asymmetric performance was caused by significant class imbalance in the training data. Only
around 22% of observations belong to the positive group thus model performance is skewed. There
are two possible approaches to rectify this issue. One is to upsample the minority class by drawing
observations with replacement until class ratio equalises. Second is to add a weight on the minority class
in the loss function effectively penalising incorrect minority class predictions more. On this particular
problem both approaches gave comparable results. However, the upsampling method was significantly
slower to train as the data size grew by around 56%, thus, the loss weighting method was chosen instead.
</div>

#### Best run

<figure>
  <img src="./Plots/run4.png" alt="fig2"/>
  <figcaption>Figure 2: Run with a positive weight in the loss function (autumn-gorge-4 from wandb)</figcaption>
</figure> <br>
<br>

<div align="justify">
Key performance metrics of the retrained LSTM using a minority class weight of 2 are displayed in Fig 2.
Similarly to the previous model, most of the training happened in the first 100 epochs. However, this
time, test data accuracy and loss curves are more volatile. The confusion matrix shows a significant
improvement for the minority class - 12p.p. increase in the true positive rate. Though that comes
at a cost of 5p.p. for the true negative rate. Classification report in Table 2 summarises the model
performance. The key metric - average F1 score - improved by 1p.p. Not a major improvement but still
welcoming. Most importantly though, the delta of the class conditional precision, recall and F1 scores
got smaller. F1 score for the minority group increased by 3p.p.
</div>

                    precision    recall  f1-score   support  

               0.0       0.91      0.90      0.90     31774  
               1.0       0.65      0.67      0.66      8925  

          accuracy                           0.85     40699  
         macro avg       0.78      0.78      0.78     40699  
      weighted avg       0.85      0.85      0.85     40699
                
    Table 2: Classification report for the best run

### Final thoughts

<div align="justify">
Performance of the LSTM ANN is not much different from what alternative methods such as XGBoost can achieve.
The default LSTM suffered from class imbalance issues which were partially rectified by including a weight
term in the loss function. The model could possibly be improved further by constructing a more complex ANN
architecture, though significant performance gains are not to be expected. There is simply not enough explanatory
power in the given features.
</div>

