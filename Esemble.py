from tensorflow import keras
import Data_load_4 as dl
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

DataX = list()
DataY = list()

dl.Data_load("None", DataX, DataY)

#dl.Data_load("Animal", DataX, DataY)

dl.Data_load("Human", DataX, DataY)

DataX = np.reshape(DataX, (int(DataX.__len__()/4), 4, 100))
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.3, random_state=777, shuffle=True)

def mlp_model():
    model = keras.Sequential()

    model.add(keras.layers.Dense(128, input_shape = (4,100)))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('sigmoid'))
    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Activation('softmax'))

    sgd = keras.layers.optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

# 서로 다른 모델을 3개 만들어 합친다

model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)

ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft')
ensemble_clf.fit(X_train, Y_train)
y_pred = ensemble_clf.predict(X_test)

print('Test accuracy:', accuracy_score(y_pred, Y_test))