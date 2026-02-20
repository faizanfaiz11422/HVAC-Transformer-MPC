import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    Note: Standardizes types to float32 for modern TF compatibility.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + tf.keras.backend.epsilon()))) * 100

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error as used in Faiz et al. (2023).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = tf.abs(y_pred - y_true)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0
    return tf.reduce_mean(numerator / (denominator + tf.keras.backend.epsilon())) * 100

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mae(y_true, y_pred):
    """Mean Absolute Error (Optimized for TensorFlow)."""
    return tf.reduce_mean(tf.abs(y_pred - y_true))
        
def mase(y_true, y_pred):
    """Mean Absolute Scaled Error."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sust = tf.reduce_mean(tf.abs(y_true[:, 1:] - y_true[:, :-1]))
    diff = tf.reduce_mean(tf.abs(y_pred - y_true))
    return diff / (sust + tf.keras.backend.epsilon())

def coeff_determination(y_true, y_pred):
    """R-squared (Coefficient of Determination)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - ss_res / (ss_tot + tf.keras.backend.epsilon()))

# convert time series to 2D data for supervised learning
def series_to_supervised(data, train_size=0.5, n_in=1, n_out=1, target_column='target', dropnan=True, scale_X=True):
    df = data.copy()

    # Make sure the target column is the last column in the dataframe
    df['target'] = df[target_column]
    df = df.drop(columns=[target_column])

    target_location = df.shape[1] - 1
    X = df.iloc[:, :]
    y = df.iloc[:, [target_location]]

    # Scale the features
    if scale_X:
        features = X[X.columns]
        scalerX = MinMaxScaler().fit(features.values)
        X[X.columns] = scalerX.transform(features.values)

    x_vars_labels = X.columns
    y_vars_labels = y.columns

    x_cols, x_names = list(), list()
    y_cols, y_names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        x_cols.append(X.shift(i))
        x_names += [('%s(t-%d)' % (j, i)) for j in x_vars_labels]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        y_cols.append(y.shift(-i))
        if i == 0:
            y_names += [('%s(t)' % (j)) for j in y_vars_labels]
        else:
            y_names += [('%s(t-%d)' % (j, i)) for j in y_vars_labels]

    # put it all together
    x_agg = pd.concat(x_cols, axis=1)
    x_agg.columns = x_names

    y_agg = pd.concat(y_cols, axis=1)
    y_agg.columns = y_names

    agg = pd.concat([x_agg, y_agg], axis=1)
    
    if dropnan:
        agg.dropna(inplace=True)

    nf = X.shape[1]
    xx = agg.iloc[:, :n_in * nf]
    yy = agg.iloc[:, -n_out:]

    split_index = int(xx.shape[0] * train_size)

    # Split into Train and temporary Test
    X_train = xx.iloc[:split_index, :]
    y_train = yy.iloc[:split_index, :]

    X_temp_test = xx.iloc[split_index:, :]
    y_temp_test = yy.iloc[split_index:, :]

    # Split temp test into CV and Test (50/50 split)
    split_cv = int(X_temp_test.shape[0] * 0.5)
    x_cv = X_temp_test.iloc[:split_cv, :]
    x_test = X_temp_test.iloc[split_cv:, :]
    y_cv = y_temp_test.iloc[:split_cv, :]
    y_test = y_temp_test.iloc[split_cv:, :]

    return X_train, y_train, x_test, y_test, x_cv, y_cv, scale_X
