import matplotlib.pyplot as plt
from model import *
from sklearn.metrics import classification_report, confusion_matrix

def visualize_results():
    plt.plot(model.history.history['accuracy'], label = 'train accuracy')
    plt.plot(model.history.history['val_accuracy'],label = 'test_accuracy')
    plt.legend()
    plt.show()

    plt.plot(model.history.history['loss'], label = 'train loss')
    plt.plot(model.history.history['val_loss'],label = 'test_loss')
    plt.legend()
    plt.show()

    #Evaluation of model to check the accuracy
    # Evaluate the model on the testing data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    predictions = model.predict(x_test[:10])
    for i in range(10):
        plt.imshow(x_test[i])
        plt.title(f"Predicted: {predictions[i]}\nTrue: {y_test[i]}")
        plt.show()