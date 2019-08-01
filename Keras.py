import tensorflow as tf;
import matplotlib.pyplot as plt;
import numpy as np;

import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';


def main():#{
    ##Get the data set
    mnist = tf.keras.datasets.mnist;
    ##get some train and test data into variables
    (xTrain,yTrain),(xTest,yTest) = mnist.load_data();
    ## normalize is to make the NN faster
    xTrain = normalize(xTrain);
    xTest = normalize(xTest);

    ##2 hidden layers
    try:#{
        model = tf.keras.models.load_model("Model/my.Model");
    #}
    except : #{
        layers = [128, 128];
        num_out = 10;
        model = getModel(layers, num_out);
        print(model)
    #}
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy']);
    model.fit(xTrain, yTrain, epochs=3);

    ##Check for generalization
    loss,acc = model.evaluate(xTest, yTest);
    print(loss, acc);
    model.save('Model/my.model');
    for i in range(1000):
        preidctions = model.predict([xTest]);
        pred = np.argmax(preidctions[i]);

        plt.figure(num="prediction: " + str(pred));
        plt.imshow(xTest[i], cmap=plt.cm.binary);
        plt.show();
#}


'''
Takes @param Tensor as data and 
'''
def normalize(tensor):#{
    return tf.keras.utils.normalize(tensor, axis=1);
    #}

'''
takes an array with number of nodes in each layer and a number of nodes for an output layer
'''
def getModel(layers,outputs:int):#{
    ##model is an ff model
    model = tf.keras.models.Sequential();
    ##add a flattening input later for the niceness
    model.add(tf.keras.layers.Flatten());
    ##make a layer of x num nodes for each layer l in the array
    for l in layers: #{
        model.add(tf.keras.layers.Dense(l,activation=tf.nn.relu));
    #}

    ## 1 output layer
    model.add(tf.keras.layers.Dense(outputs, activation=tf.nn.softmax));
    return  model;
    #}

def trainModel(model,xTrain, yTrain):#{
    ##training
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy']);
    model.fit(xTrain, yTrain, epochs=3);
    #}















if __name__ == "__main__":
    main()