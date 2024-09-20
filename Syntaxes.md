This MD walks users through the lastest SCENet(v3) Framework syntaxes
* Requirements for Input and Output:
  >> The Input must be a 2d array, with each column represents a sample.
  >> 
  >> The Output is an 1d array, every index represents an actual output value.
* To init a Neural network:
  >> nn = NeuralNetwork()
* (Optional) Setup Learning_rate(Alpha) and/or Batch_size, unitGradient makes all generated gradients to be of length 1
  >> nn.set(alpha=0.01, batch_size=32, unitGradient=True) <- Default values
* (Optional) Safety mode(Auto adjusting learning rate)
  >> nn.safety(whenDropBy = 0.01, reduceLRBy = 0.05, stoppingLR = 0.0001) ##reduce learning rate by 5% for when accuracy drops by 1% and stop Training if lr <= 0.0001
  >> 
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
  >> A new feature from v3: If trainings happen on a Notebook(Jupyer, Collab, etc.) running the cell which has fit() function tells the model to train itself again using the most recent Weights and Biases. ! Do not regenerate a new object nor pass in new X-train and Y-train.
  >>
  >> If wanted, users can change learning rate (alpha) / Not batch_size as well.
  >> 
  >> Thus, if users find the need to train the model more without starting over, they only need to run the fit() again and again.
* (Optional) View description of Neural Network structure(After Training is Done)
  >> nn.describe()
* Predict and Accuracy
  >> predictions = nn.predict(x_validate)
  >> 
  >> nn.get_accuracy(predictions, y_validate)
