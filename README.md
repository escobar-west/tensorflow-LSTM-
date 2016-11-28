# tensorflow-LSTM-
This is a four layer LSTM neural network I coded using TensorFlow to predict time series data of ozone levels in the atmosphere. The first hidden layer consists of LSTM cells, the second and third layers are fully connected hidden layers, and the final layer is the output layer. It trains using back propogation and a mean square error function.

The code is pretty straight forward. The main program is doubleLSTM.py, which sets up the TensorFlow graph and runs the session. tsModule.py is a lightweight class that creates a data object and feeds data to the neural net. The raw data is stored in Ozone.csv.

Just save all of the files in one folder and run the program doubleLSTM.py. Of course, you will need TensorFlow installed in your computer (and matplotlib.pyplot for plotting some graphs). The hyperparameters such as batch size and number of nodes can be changed in the main program.
