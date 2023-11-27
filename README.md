# Recurrent-Neural-Networks-for-Stock-price-prediction

This project focuses on predicting the next day’s stock price based on the previous three days' data, including Volume, Open, High, and Low prices. It leverages Recurrent Neural Networks (RNN), specifically Long Short-Term Memory (LSTM) networks, due to their effectiveness in handling time-series data with long and short-term dependencies.

**Dataset** <br>
The dataset used (q2_dataset) includes the following columns: Date, Close/Last, Volume, Open, High, and Low. The primary task is to predict the "Open" price for the next day using the past three days' data.

**Data Preparation** <br>
Feature Engineering: The dataset is transformed so that each sample contains data from 3 consecutive days, resulting in 12 features.
Target Variable: The "Open" price of the fourth day is used as the target.
Splitting: The dataset is randomized and split into 70% training and 30% testing sets, saved as train_data_RNN.csv and test_data_RNN.csv, respectively.

**Preprocessing** <br>
The training dataset is loaded and separated into features (x_train) and target (y_train).
Standard scaling normalization is applied to the features, normalizing them to a mean of 0 and a standard deviation of 1.
The data is reshaped to fit the LSTM model's input requirements.

**Model Architecture** <br>
The network consists of two LSTM layers with 48 and 82 units, followed by a fully connected output layer with linear activation.
The model has a total of 52,651 parameters.
It is trained for 155 epochs with a batch size of 8.

![image](https://github.com/bhupeshdod/Recurrent-Neural-Networks-for-Stock-price-prediction/assets/141383468/a7f245d5-3fcb-4b76-a52c-3e1c5db1c55a)

**Results** <br>
Training Loss: Converged to approximately 10.
Validation Loss: Converged to approximately 9.
Training MAE: Approximately 2.0.
Validation MAE: Approximately 2.06.

![image](https://github.com/bhupeshdod/Recurrent-Neural-Networks-for-Stock-price-prediction/assets/141383468/56261ff8-2350-493a-a165-44eb56e28ef1)

**Testing** <br>
MSE on Test Dataset: Approximately 13.5.
MAE on Test Dataset: Approximately 2.16.
Observations: Most predictions align closely with true values, with some deviations.

**Additional Experiments** <br>
Experimenting with the number of days used for features showed variable results. Increasing days can provide more historical context but may also introduce complexity and noise
