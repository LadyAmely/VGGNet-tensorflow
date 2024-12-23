import tensorflow as tf
def VGGNet(input_shape=(224, 224, 3), num_classes=1000):

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


model = VGGNet(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
