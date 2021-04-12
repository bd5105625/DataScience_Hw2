import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Flatten, TimeDistributed, RepeatVector, Input, Dropout
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def lstm_stock_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model


def buildtrain(training_data):
    passday = 10
    X_train, Y_train = [], []
    for i in range(training_data.shape[0]-10):
        X_train.append(training_data.iloc[i:i+passday,0].values)
        Y_train.append(training_data.iloc[i+passday,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    print(X_train.shape , Y_train.shape)
    print(X_train)
    X_train = np.reshape(X_train , (X_train.shape[0], X_train.shape[1], 1))
    return (X_train), (Y_train)

def trade(training_data , testing_data):
    print(training_data)
    print(testing_data)
    X_train, Y_train = buildtrain(training_data)
    print(X_train.shape , Y_train.shape)


    dataset_total = pd.concat((training_data.iloc[-100:,0] , testing_data.iloc[:,0]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(testing_data.iloc[:,0]) - 10:].values
    X_test = []
    print(len(inputs))
    for i in range(100):
            # X_test.append(testing_data.iloc[i:i+10,0].values)
            X_test.append(dataset_total.iloc[i:i+10].values)
    X_test = np.array(X_test)
    print(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    print(X_test.shape)

    
    model = lstm_stock_model(X_train.shape)
    callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
    history = model.fit(X_train, Y_train, epochs=30, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)

    # model.save("model.h5")
    # model = keras.models.load_model("model.h5")
    predict_price = model.predict(X_test)
    predict_price = predict_price[-20:]
    # plt.figure(1)
    # plt.plot(predict_price[-20:], 'blue')
    answer = []
    unit = 0
    for i in range(19):
        if predict_price[i] > predict_price[i+1]: # slump going to sell
            if unit == 0: # no stock
                answer.append(-1) # sell or short
                unit = -1
                print('1 ', predict_price[i] , predict_price[i+1])
            elif unit == 1: # one stock 
                answer.append(-1)
                unit = 0 # sell stock and return to 0
                print('2 ', predict_price[i] , predict_price[i+1])
            elif unit == -1: # already shorted a stock  
                answer.append(0) # no action
                print('3 ', predict_price[i] , predict_price[i+1])
        elif predict_price[i] < predict_price[i+1]: # elevate going to buy
            if unit == 0:
                answer.append(1)
                unit = 1
                print('4 ', predict_price[i] , predict_price[i+1])
            elif unit == 1:
                answer.append(0) # no action
                print('5 ', predict_price[i] , predict_price[i+1])
            elif unit == -1:
                answer.append(1) # buy
                unit = 0
                print('6 ', predict_price[i] , predict_price[i+1])
        else:
            answer.append(0)
    output = pd.DataFrame(answer)
    print(output)
    # f = open("test.txt" , "w")
    # f.write(predict_price)
    # f.close()
    # plt.figure(1)
    # plt.plot(openprice1, 'blue')
    # plt.figure(2)
    # plt.plot(openprice2, 'blue')
    # plt.show()
    return output

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = pd.read_csv(args.training, header=None)
    
    testing_data = pd.read_csv(args.testing, header=None)
    output = trade(training_data, testing_data)
    output.to_csv(args.output , header=False, index=False)
    # plt.show()