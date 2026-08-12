import mnist_loader
import network1
import matplotlib.pyplot as plt
import numpy as np
training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

net = network1.Network([784, 30, 10])

net.SGD(training_data, 30, 10,  3.0, test_data=test_data)


plot, ax = plt.subplots(figsize=(8,8))


es = np.arange(30)
print(len(net.loss))

ax.scatter(es, net.loss)
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
plt.show()
