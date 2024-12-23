from models.vggnet import VGGNet
from dataset.dataset_loader import load_and_preprocess_data
from utils.visualization import plot_training_results

def train_model():
    train_generator, test_generator = load_and_preprocess_data()

    model = VGGNet(input_shape=(224, 224, 3), num_classes=10)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=10)

    plot_training_results(history)
    model.save('../saved_models/vggnet.h5')

if __name__ == "__main__":
    train_model()
