В этом репозитории мы сравним точность распознавания тестового набора цифр MNIST.
Для это мы описали для модели нейронной сети: MLP (Net) и CNN + MLP (ConvNet).
Набор цифр в файле "data.bin" имеет вид: (x_train, y_train), (x_test, y_test)

Реализация структуры второй модели взаимствована [отсюда](https://www.educba.com/pytorch-conv2d/).



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
