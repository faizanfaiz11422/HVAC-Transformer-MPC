from datetime import datetime
from time import time
import json
import logging

import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Flatten
from keras.layers import MaxPooling1D, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.regularizers import l1

from kerastuner.tuners import RandomSearch

from sklearn.metrics import r2_score


from utils import rmse, coeff_determination, smape


class CNN(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self):
        """ Initialization of the object
        """

        with open("parameters.json") as f:
            parameters = json.load(f)


        # Get model hyperparameters
        self.look_back = parameters["look_back"]
        self.n_features = parameters["n_features"]
        self.horizon = parameters["horizon"]

        # Get directories name
        self.log_dir = parameters["log_dir"]
        self.checkpoint_dir = parameters["checkpoint_dir"]

        self.head_size=256
        self.num_heads=4
        self.ff_dim=4
        self.num_transformer_blocks=4
        self.mlp_units=[128]
        self.mlp_dropout=0.4
        self.dropout=0.25


    def cnn_lstm(self,
        inputs):

        model = Sequential()
        model.add(Conv1D(64,
                        kernel_size=32,
                        padding='same',
                        activation='tanh',
                        strides=1))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(128,
                        kernel_size=16,
                        padding='same',
                        activation='tanh',
                        strides=1))
        # model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=4))
        # model.add(TimeDistributed((Dropout(0.2))))
        model.add(LSTM(units = 60, return_sequences=True, 
                      ))
        model.add(LSTM(units = 10))
        model.add(Flatten())

        model.compile(optimizer='adam', loss = ['mse'], metrics=[rmse, 'mae', smape, coeff_determination])

        return model
        # model.add(Dropout(0.2))
        # model.add(Dense(units = 1, activity_regularizer=l1(0.015)))
        # model.summary()
        # optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
        # model.compile(optimizer = optimizer , loss = 'mse')    


    # def build(self):
    #     """ Build the model architecture
    #     """

    #     inputs = keras.Input(shape=(self.look_back, self.n_features))
    #     x = inputs
    #     for _ in range(self.num_cnn_filters):
    #         x = self.CNN

    #     x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    #     for dim in self.mlp_units:
    #         x = layers.Dense(dim, activation="relu")(x)
    #         x = layers.Dropout(self.mlp_dropout)(x)

    #     # output layer
    #     outputs = layers.Dense(self.horizon)(x)

    #     return keras.Model(inputs, outputs)

    # def restore(self,
    #     filepath):
    #     """ Restore a previously trained model
    #     """

    #     # Load the architecture
    #     self.best_model = load_model(filepath, custom_objects={'smape': smape,
    #                                                      #'mape': mape,
    #                                                      'rmse' : rmse,
    #                                                      'coeff_determination' : coeff_determination})

    #     ## added cause with TF 2.4, custom metrics are not recognize custom metrics with only load-model
    #     self.best_model.compile(
    #         optimizer='adam',
    #         loss = ['mse'],
    #         metrics=[rmse, 'mae', smape, coeff_determination])


    def train(self,
        X_train,
        y_train, x_cv, y_cv,
        epochs=1,
        batch_size=64):
        """ Training the network
        :param X_train: training feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_train: training target vectors
        :type 2-D Numpy array of float values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """
              
        input_shape=(1, self.look_back, self.n_features)
        inputs = input_shape
        self.model = self.cnn_lstm(inputs)
        self.model.build(inputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss = ['mse'],
                           metrics=[rmse, 'mae', smape, coeff_determination],
                           )
        print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        filepath = self.checkpoint_dir+"/Transformer.best"+datetime.now().strftime('%d%m%Y_%H:%M:%S')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             #validation_split=0.2,
                             verbose=1, validation_data = (x_cv, y_cv),
                             callbacks=[early_stopping_monitor, checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])

        return callback_history

    def evaluate(self,
        X_test,
        y_test):
        """ Evaluating the network
        :param X_test: test feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_test: test target vectors
        :type 2-D Numpy array of int values
        :return  Evaluation losses
        :rtype 5 Float tuple
        :raise -
        """

        y_pred = self.model.predict(X_test)

        # Print accuracy if ground truth is provided
        """
        if y_test is not None:
            loss_ = session.run(
                self.loss,
                feed_dict=feed_dict)
        """

        _, rmse_result, mae_result, smape_result, _ = self.model.evaluate(X_test, y_test)

        r2_result = r2_score(y_test.flatten(),y_pred.flatten())

        return _, rmse_result, mae_result, smape_result, r2_result, y_pred

 
