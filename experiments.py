from models import *
from data import *
import argparse
from sklearn.model_selection import train_test_split
import datetime

#Argument parser 
parser = argparse.ArgumentParser(description='Specify model hyperparameters and data path')

#Model args
parser.add_argument('--conv_layers', type=int, help="Specify integer number of convolutional layers")
parser.add_argument('--filters', nargs='+', type=int, help="List containing integer number of convolutional filters for each layer")
parser.add_argument('--kernel_size', nargs='+', type=int, help="List containing integer size of kernels in each convolutional layer")
parser.add_argument('--dilation_rate', type=int, help="Specify integer dilation rate of successive convolutional layers")
parser.add_argument('--gru_cells', type=int, help="Specify integer number of GRU cells in RNN layer")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs to train the model")
parser.add_argument('--data_path', help="Specify path of saved data")

#Collect args
args = parser.parse_args()

#Building model
model = GWN(conv_layers=args.conv_layers,
            filters=args.filters,
            kernel_size=args.kernel_size,
            dilation_rate=args.dilation_rate,
            gru_cells=args.gru_cells)

#Loading data
data = np.load(args.data_path, allow_pickle=True)

#Separating data
data = data[0]
data_labels = data[1]
noise = data[2]
noise_labels = data[3]

#Combining data
X, y = combine_data(data, noise, data_labels, noise_labels)

#Shuffling data with fixed seed for reproducability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)

#Training model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=args.epochs)

#Test the model
results = model.evaluate(X_test, y_test, verbose=2)
test_loss = results[0]
test_accuracy = results[1]

#Logging results
file = str(datetime.datetime.now()).replace(" ", "-")
experiment_output = open(file, 'w+')
logs = [
    f"Date: {datetime.datetime.now()}",
    f"conv_layers: {args.conv_layers}",
    f"filters: {args.filters}",
    f"kernel_size: {args.kernel_size}",
    f"dilation_rate: {args.dilation_rate}",
    f"gru_cells: {args.gru_cells}",
    f"epochs: {args.epochs}",
    f"data: {args.data_path}",
    f"Test loss: {test_loss}",
    f"Test accuracy: {test_accuracy}"
]
experiment_output.writelines(logs)
