Feb 16, 2024_SCENet: The first Deep Learning Framework by BilldaLab
+ Pros:
   Friendly syntaxes as the goal for this Project.
   Users can easily add as many hidden layers and neurons as desired.
   Available activation functions: ReLU, Sigmoid, TanH, Identity.
- Cons:
   High additions on hidden layer and/or neuron may lead to exhausting Runtime.
   This Neural Network framework only supports Sparse Categorical Cross Entropy loss
    --> That is why its name is SCENet (Sparse Categorical crossEntropy Net)

Feb 20, 2024_SCENet2: Mini-batch Gradient Descent is built-in
- Training speed has increased substantially.

Feb 25, 2024_SCENet3: Train on any dataType of Labels and Auto-fixed learning rate
- In the previous versions, users need to translate each output possibility to be a digit which causes a tedious process. In this version, users can train the model on almost any type of Labels, from numbers to categories. Although it takes some time at the 1st epoch, but it is more efficient.
- Users are now able to turn on safety mode. If the training accuracy decreases significantly, SCENet3 automatically adjusts the learning rate. At the end of the trainings, users can see how many times their models drop in accuracy as well as the final adjusted learning rate.
