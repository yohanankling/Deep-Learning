from model import build_model, load_cifar10, build_model_old
from gui import plot_results2
from logisticRegression import logistic_regression
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras as keras

# Declaration of some global variables:
EPOCHS = 15

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_cifar10()

    # Build the models
    model = build_model()
    old_model = build_model_old()

    # Learning Rate Scheduler
    def lr_schedule(epoch):
        return 0.001 * 0.9 ** epoch


    # Train the model
    results = model.fit(train_images, train_labels, validation_split=0.1, epochs=EPOCHS,
                        callbacks=[LearningRateScheduler(lr_schedule)])

    # Train the old model
    # old_results = old_model.fit(train_images,
    #                             train_labels,
    #                             epochs=EPOCHS,
    #                             validation_split=0.2,
    #                             callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # Evaluate the models on test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    # old_test_loss, old_test_acc = old_model.evaluate(test_images, test_labels)

    print("Test accuracy for the new model:", test_acc)
    # print("Test accuracy for the old model:", old_test_acc)

    plot_results2(results.history['accuracy'],
                  results.history['val_accuracy'],
                  "Training Accuracy", "Validation Accuracy", "Epochs", "Accuracy")

    plot_results2(results.history['loss'],
                  results.history['val_loss'],
                  "Training Loss", "Validation Loss", "Epochs", "Loss")

    # plot_results2(
    #     results.history['accuracy'],
    #     old_results.history['accuracy'],
    #     "New Model Training Accuracy", "Old Model Training Accuracy", "Epochs", "Accuracy")
    #
    # plot_results2(
    #     results.history['loss'],
    #     old_results.history['loss'],
    #     "New Model Training Loss", "Old Model Training Loss", "Epochs", "Loss")

    # # Logistic Regression
    # logistic_regression_accuracy = logistic_regression(train_images, train_labels, test_images, test_labels)
