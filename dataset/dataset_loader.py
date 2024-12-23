from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=32)

    return train_generator, test_generator