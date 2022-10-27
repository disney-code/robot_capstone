import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
def load_cifar10():
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
    return (x_train, y_train), (x_vali, y_vali),(x_test, y_test)