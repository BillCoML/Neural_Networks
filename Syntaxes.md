This MD walks users through JoyBillv2 Framework syntaxes
* Requirements for Input and Output:
  >> The Input must be a 2d array, with each column represents a sample.
  >> 
  >> The Output is an 1d array, every index represents an actual output value.
* To init a Neural network:
  >> nn = NeuralNetwork()
* (Optional) Setup Learning_rate(Alpha) and/or Batch_size
  >> nn.set(alpha=0.01, batch_size=32) <- Default values
* Add Layers (Each with amount of hidden neurons and Activation function)
  -> relu / sigmoid / tanh / identity
  >> nn.addLayer(10, 'relu')
  >> 
  >> nn.addLayer(15, 'sigmoid')
  >> 
  >> ...
* Add Output Layer (amount of hidden neurons without Activation function)
  >> nn.addLayer(10)
* Start Training (required fields: x_train, y_train, epochs)
  >> nn.fit(x_train, y_train, epochs = 5)
  >> 
  >> A new feature from v3: If trainings happen on a Notebook(Jupyer, Collab, etc.) running the cell which has fit() function tells the model to train itself using the most recent Weights and Biases. ! Do not regenerate a new object nor pass in new X-train and Y-train.
  >> Thus, if users find the need to train the model more without starting over, they only need to run the fit() again and again.
* (Optional) View description of Neural Network structure(After Training is Done)
  >> nn.describe()
* Predict and Accuracy
  >> predictions = nn.predict(x_validate)
  >> 
  >> nn.get_accuracy(predictions, y_validate)
