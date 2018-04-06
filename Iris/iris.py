import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Fix the random seed
seed = 7
numpy.random.seed(seed)

# Load the dataset
dataframe = read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Encode the class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Conver the integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

# Use Keras to create the neural network
# We will have:
# - Input layer
# - One Hidden Layer with 8 neurons
# - Output Layer with 3 output values (for each category)
def baseline_model():
	# Create the model
	model = Sequential()
	# Use Rectified Linear Units for activation
	# for good results and better speed
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Use the Keras model in Scikit
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

# Evaluate with kFold
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

