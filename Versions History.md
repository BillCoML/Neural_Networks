Feb 16, 2024_JoyBillv1: The first Deep Learning Framework by BilldaLab
+ Pros:
   Friendly syntaxes as the goal for this Project.
   Users can easily add as many hidden layers and neurons as desired.
   Available activation functions: ReLU, Sigmoid, TanH, Identity.
- Cons:
   High additions on hidden layer and/or neuron may lead to exhausting Runtime.

Feb 20, 2024_JoyBillv2: Mini-batch Gradient Descent is built-in
- Training speed has increased substantially.

Feb 25, 2024_JoyBillv3: Train on any dataType of Labels and Auto-fix learning rate
- In the previous versions, users need to translate each output possibility to be digit which causes a tedious process.
!!! Using this version, users can feed the model almost any type of Labels, from numbers to categories. Although it takes some time, but it is easier.
- Now users are able to turn on safety mode. If the training accuracy decreases significantly, JoyBillv3 auto adjusts the learning rate. At the end of the trainings, users can see how many times their model drops in accuracy as well as the final adjusted learning rate.
