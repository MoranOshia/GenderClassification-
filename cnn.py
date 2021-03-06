import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def cnn_(x_train, x_test, y_train, y_test,X):

    inputs = tf.keras.Input(shape=(X.shape[1],))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    print("Evaluate on test data for CNN: ")
    model.evaluate(x_test, y_test)

   
def cnn2D(x_train, x_test, y_train, y_test,X,y):
    
    X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype=np.float, maxlen=25, padding='post')
    X = X.reshape(-1, 5, 5)
    X = np.expand_dims(X, axis=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

    x = tf.keras.layers.Conv2D(16, 2, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 1, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    print("Evaluate on test data for 2D CNN: ")
    model.evaluate(X_test, y_test)