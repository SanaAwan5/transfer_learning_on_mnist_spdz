from keras.utils import to_categorical
from keras.datasets import mnist
from emnist import list_datasets, extract_training_samples, extract_test_samples

#pip install emnist
def preprocess_data(dataset):
    
    (x_train, y_train), (x_test, y_test) = dataset
    
    # NOTE: this is the shape used by Tensorflow; other backends may differ
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    y_train = to_categorical(y_train, 27)
    y_test  = to_categorical(y_test, 27)
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_data2(dataset):
    
    (x_train, y_train), (x_test, y_test) = dataset
    
    # NOTE: this is the shape used by Tensorflow; other backends may differ
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    

    x_train_public = x_train[y_train < 5]
    y_train_public = y_train[y_train < 5]
    x_test_public  = x_test[y_test < 5]
    y_test_public  = y_test[y_test < 5]
    public_dataset = (x_train_public, y_train_public), (x_test_public, y_test_public)

    x_train_private = x_train[y_train >= 5]
    y_train_private = y_train[y_train >= 5] - 5
    x_test_private  = x_test[y_test >= 5]
    y_test_private  = y_test[y_test >= 5] - 5
    private_dataset = (x_train_private, y_train_private), (x_test_private, y_test_private)
    
    return preprocess_data(public_dataset), preprocess_data(private_dataset)

def load_emnist():
    
    (x_train, y_train) = extract_training_samples('letters')
    (x_test, y_test) = extract_test_samples('letters')

    public_dataset = (x_train, y_train), (x_test, y_test)
    (x_trainm, y_trainm), (x_testm, y_testm) = mnist.load_data()
    x_train_private = x_trainm[y_trainm >= 5]
    y_train_private = y_trainm[y_trainm >= 5] - 5
    x_test_private  = x_testm[y_testm >= 5]
    y_test_private  = y_testm[y_testm >= 5] - 5
    private_dataset = (x_trainm, y_trainm), (x_testm, y_testm)

    return preprocess_data(public_dataset),preprocess_data2(private_dataset)
    
   
    