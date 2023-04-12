from sklearn.model_selection import train_test_split
from load_data import load_data
from model import *
from visualize_data import *

if __name__ == '__main__':
    data, targets = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

    model.fit(x_train, y_train,batch_size=32,epochs=5,validation_data=(x_test, y_test))

    visualize_results()