The English translation is below.

В этом репозитории мы сравним точность распознавания тестового набора цифр MNIST.
Для это мы описали для модели нейронной сети: MLP (Net) и CNN + MLP (ConvNet).
Набор цифр в файле "data.bin" имеет вид: (x_train, y_train), (x_test, y_test).
Обучение нейронных сетей производится с помощью модуля PyTorch

Реализация структуры второй модели взаимствована [отсюда](https://www.educba.com/pytorch-conv2d/).
Сверточная нейронная сеть дает точность на 1% большую, чем простой многослойный 
персептрон, уступая во времени обучения в 6 раз.

Результат выполнения представлен ниже (время в секундах)

English
In this repository, we will compare the recognition accuracy of the MNIST test set of digits. To do this, we have described for the neural network model: MLP (Net) and CNN + MLP (Convention). The set of digits in the "data.bin" file has the form: (x_train, y_train), (x_test, y_test). Neural networks are trained using the Pwtorch module

The implementation of the structure of the second model is borrowed from [here](https://www.educba.com/pytorch-conv2d/). A convolutional neural network provides 1% greater accuracy than a simple multilayer perceptron, yielding 6 times in training time.

The result of the execution is shown below (time in seconds)
```
Training of MLP...
   ...
   Epoch: 9, accuracy: 99.51%, time: 110.72
   Survey accuracy: 98.11%, time: 2.08
   Overall time: 1131.75
Training of CNN + MLP...
   ...
   Epoch: 9, accuracy: 99.34%, time: 637.07
   Survey accuracy: 99.26%, time: 28.40
   Overall time: 6604.93
```
