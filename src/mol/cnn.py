"""
Estágio Learn: a CNN do artigo (1 convolução de 32 filtros, 1 max-pooling e 1 camada
densa com (entradas + saídas) / 2 neurônios, Adam, até 100 épocas).

MOLCNNClassifier junta Map, Optimize e CNN num estimador scikit-learn, para que a
validação cruzada refaça todas as etapas dentro de cada fold.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

import keras
from keras import layers

from .algorithm import MapOptimizeLearn


def build_cnn(input_shape, n_hidden, n_classes, learning_rate):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(n_hidden, activation='relu'),
        layers.Dense(1, activation='sigmoid') if n_classes == 2
        else layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy',
    )
    return model


class MOLCNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Aceita os parâmetros de MapOptimizeLearn e os da CNN. Com early_stopping=True,
    validation_fraction do treino vira validação e o treino para depois de `patience`
    épocas sem melhora.
    """

    def __init__(self, map_strategy='euclidean', optimizer='pso', n_population=50,
                 max_evaluations=500, sample_order='class', min_pixels=2,
                 scaler='standard', learning_rate=1e-3, epochs=100, batch_size=32,
                 early_stopping=False, validation_fraction=0.15, patience=15,
                 random_state=42, verbose=0):
        self.map_strategy = map_strategy
        self.optimizer = optimizer
        self.n_population = n_population
        self.max_evaluations = max_evaluations
        self.sample_order = sample_order
        self.min_pixels = min_pixels
        self.scaler = scaler
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        y_idx = np.searchsorted(self.classes_, y)

        self.mol_ = MapOptimizeLearn(
            map_strategy=self.map_strategy, optimizer=self.optimizer,
            n_population=self.n_population, max_evaluations=self.max_evaluations,
            sample_order=self.sample_order, min_pixels=self.min_pixels,
            scaler=self.scaler, random_state=self.random_state, verbose=self.verbose,
        ).fit(X, y_idx)
        images = self.mol_.transform(X)[..., np.newaxis]

        # o grid search treina centenas de redes; sem isso a memória cresce a cada ajuste
        keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        n_pixels = int(np.prod(self.mol_.image_shape_))
        n_out = 1 if len(self.classes_) == 2 else len(self.classes_)
        n_hidden = max(1, int(np.ceil((n_pixels + n_out) / 2)))
        self.model_ = build_cnn(images.shape[1:], n_hidden, len(self.classes_), self.learning_rate)

        fit_kwargs = dict(epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        if self.early_stopping:
            X_tr, X_val, y_tr, y_val = train_test_split(
                images, y_idx, test_size=self.validation_fraction,
                stratify=y_idx, random_state=self.random_state)
            fit_kwargs.update(
                validation_data=(X_val, y_val),
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.patience, restore_best_weights=True)])
        else:
            X_tr, y_tr = images, y_idx

        self.history_ = self.model_.fit(X_tr, y_tr, **fit_kwargs).history
        return self

    def predict_proba(self, X):
        images = self.mol_.transform(X)[..., np.newaxis]
        p = self.model_.predict(images, verbose=0)
        if p.shape[1] == 1:
            p = np.hstack([1 - p, p])
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def transform_to_images(self, X):
        return self.mol_.transform(X)
