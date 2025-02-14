import tensorflow as tf


class VGGNet(tf.keras.Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(VGGNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Blok 1
        self.conv1_1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # Blok 2
        self.conv2_1 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # Blok 3
        self.conv3_1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # Blok 4
        self.conv4_1 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # Blok 5
        self.conv5_1 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # Warstwy Dense
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.output_layer(x)
