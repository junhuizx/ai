import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, AveragePooling2D, Convolution2D, Activation
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.callbacks import Callback
from keras.optimizers import SGD


class LossHistory(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.losses = []
        self.accuracies = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


history = LossHistory()

data = pd.read_csv("train.csv")
data = data.sample(n=10000, replace=False)
digits = data[data.columns.values[1:]].values
labels = data.label.values

train_digits, test_digits, train_labels, test_labels = train_test_split(digits, labels)

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(
    Convolution2D(nb_filter=32, nb_row=3, nb_col=3, dim_ordering="th", border_mode="same", bias=False, init="uniform"))
model.add(AveragePooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(
    Convolution2D(nb_filter=64, nb_row=3, nb_col=3, dim_ordering="th", border_mode="same", bias=False, init="uniform"))
model.add(AveragePooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Flatten())
model.add(Dense(output_dim=1000, activation="sigmoid"))
model.add(Dense(output_dim=1000, activation="sigmoid"))
model.add(Dense(output_dim=10, activation="linear"))

with open("digits_model.json", "w") as f:
    f.write(model.to_json())

plot(model, to_file="digits_model.png", show_shapes=True)

model.to_json()

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])

model.fit(train_digits, train_labels_one_hot, batch_size=32, nb_epoch=10, callbacks=[history])

model.save_weights("digits_model_weights.hdf5")

predict_labels = model.predict_classes(test_digits)

print(classification_report(test_labels, predict_labels))
print(accuracy_score(test_labels, predict_labels))
print(confusion_matrix(test_labels, predict_labels))
