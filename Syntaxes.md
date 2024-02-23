This MD walks users through JoyBill Framework syntaxes
* To init a Neural network:
  >> nn = NeuralNetwork()
* (Optional) Setup Learning_rate(Alpha) and/or Batch_size
  >> nn.set(alpha=0.01, batch_size=32) <- Default values
* Add Layers (Each with amount of hidden neurons and Activation function)
  -> relu / sigmoid / tanh / identity
  >> nn.addLayer(10, 'relu')
  >> nn.addLayer(15, 'sigmoid')
  >> ...
* Add Output Layer (amount of hidden neurons without Activation function)
  >> nn.addLayer(10)
* Start Training (required fields: x_train, y_train, epochs)
  >> nn.fit(x_train, y_train, epochs = 5)
* (Optional) View description of Neural Network structure(After Training is Done)
  >> nn.describe()
* Predict and Accuracy
  >> predictions = nn.predict(x_validate)
  >> nn.get_accuracy(predictions, y_validate)
