# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0325

"""
Warper script for the k2arm framework to
create .h and .c source files which represent a pre trained keras model.
This code can be used to run the keras model with the ARM CMSIS-NN functions
on an embedded target such as the Cortex-M4

@author: Raphael Zingg zing@zhaw.ch
@copyright: 2019 ZHAW / Institute of Embedded Systems
"""
import keras as k
from k2arm import k2arm

# --------------------------------------------------------------------------------------------------
#                                           settings
# --------------------------------------------------------------------------------------------------
modelPath = '../../models/modelLar/model.keras' # chose the model (modelMin, modelLar, modelMid)
qFormat = 15 # set to 7 or 15

# --------------------------------------------------------------------------------------------------
#                                      load and scale data
# --------------------------------------------------------------------------------------------------
mnist = k.datasets.mnist
(x_train, y_train), (x_test_int, y_test) = mnist.load_data()
x_test = x_test_int / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28)
mnistEvalData = [x_test, y_test]

# --------------------------------------------------------------------------------------------------
#                                      load pre-trained model
# --------------------------------------------------------------------------------------------------
classifier = k.models.load_model(modelPath)

# --------------------------------------------------------------------------------------------------
#                                      create k2arm parser
# --------------------------------------------------------------------------------------------------
parser = k2arm(outputFilePath='../target/', fixPointBits=qFormat,
               model=classifier, evalData=mnistEvalData)

# --------------------------------------------------------------------------------------------------
#                      parse the model from keras to CMSIS-NN parameters
# --------------------------------------------------------------------------------------------------
parser.quantizeWeights()
parser.findOutputFormat()

# --------------------------------------------------------------------------------------------------
#                                      write the C-code files
# --------------------------------------------------------------------------------------------------
parser.storeWeights()
parser.storeDimension()
parser.storeOutShiftParams()
parser.storeNetFunction()
parser.storeBitSize()
print('Model successfully parsed and C-code generated!')
