# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0325

"""
compare the predictions of MNSIT pictures of the original net and the net on the
embedded target, using UART communication

!! adjust serDev according to your serial adapter !!

@author: Raphael Zingg zing@zhaw.ch
@copyright: 2019 ZHAW / Institute of Embedded Systems
"""
import struct
import keras as k

from M4Driver import M4Driver

# --------------------------------------------------------------------------------------------------
#                                             settings
# --------------------------------------------------------------------------------------------------
serDev = '/dev/ttyUSB1'
modelPath = 'models/modelLar/model.keras'

# --------------------------------------------------------------------------------------------------
#                           load data, scale into float and reshape as vector
# --------------------------------------------------------------------------------------------------
mnist = k.datasets.mnist
(x_train, y_train), (x_test_int, y_test) = mnist.load_data()
x_test = x_test_int / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28)

# --------------------------------------------------------------------------------------------------
#                                        load and evaluate model
# --------------------------------------------------------------------------------------------------
cModel = k.models.load_model(modelPath)
accHost = cModel.evaluate(x_test, y_test, verbose=0)

# --------------------------------------------------------------------------------------------------
#                                open serial connection to the M4 board
# --------------------------------------------------------------------------------------------------
m4d = M4Driver()
m4d.openSerial(serDev)

# --------------------------------------------------------------------------------------------------
#                                 get the predictions from the board
# --------------------------------------------------------------------------------------------------
print('Comparing ' + str(x_test_int.shape[0]) + ' MNIST predictions, this takes a while')
wrongPred = 0
for i in range(0, x_test_int.shape[0]):
    cPred = m4d.predict(x_test_int[i].reshape(1, 28*28))
    cPred = struct.unpack('1B', cPred)[0]
    print(cPred, y_test[i])
    if cPred != y_test[i]:
        wrongPred = wrongPred + 1

# --------------------------------------------------------------------------------------------------
#                                     calculate the accuracy
# --------------------------------------------------------------------------------------------------
accM4 = str(100 - (wrongPred / x_test.shape[0]) * 100)
accHost = str(accHost[1] * 100)
print('Compared model:' + modelPath)
print('Accuracy host:' + accHost +'%\n' + 'Accuracy target:' + accM4 + '%')
