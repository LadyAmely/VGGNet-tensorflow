import os
import sys
models_dir = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(models_dir)
from models.vggnet import VGGNet
from dataset.dataset_loader import load_and_preprocess_data
import tensorflow as tf

def train_model():

    train_ds, test_ds = load_and_preprocess_data(batch_size=32)
    model = VGGNet(input_shape=(64, 64, 3), num_classes=10)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-3
    )


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )


    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=4,
        callbacks=[early_stopping, lr_scheduler]
    )


    model.export('../saved_models/vggnet')
    model.save('../saved_models/vggnet.keras')
    model.save('../saved_models/vggnet.h5')



if __name__ == "__main__":
    train_model()