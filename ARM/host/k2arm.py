# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0325
""" Class to parse a keras model into a .h and .c file such that the ARM CMSIS NN library
functions can be used to run the model on an embedded micro controller such as a Cortex-M4.

More information about the ARM CMSIS:
    https://www.keil.com/pack/doc/CMSIS/NN/html/index.html
    
More information about keras:
    https://keras.io/

References:
    [1]: KEIL CMSIS Documentation,
         https://www.keil.com/pack/doc/CMSIS/DSP/html/group__float__to__x.html

    [2]: Tensorflow documentation,
         https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_args

    [3]: liangzhen-lai, ARM-CMSIS Github, https://github.com/ARM-software/CMSIS_5/issues/327

@author: Raphael Zingg zing@zhaw.ch
@copyright: 2018 ZHAW / Institute of Embedded Systems
"""
import datetime
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Activation, Lambda
from keras.models import Model


class k2arm:
    def __init__(self, outputFilePath, fixPointBits=7, model=None, evalData=None):
        self.outputFilePath = outputFilePath
        self.fixPointBits = fixPointBits
        self.model = model
        self.evalData = evalData
        self.numberOfDenselayers = len(self.getDenseLayersInformation())
        self.qWeightsAndBias = []
        self.weightDecBits = []
        self.intWeightsAndBias = []
        self.bestIntBits = []

# --------------------------------------------------------------------------------------------------
#                                      functions to quantize weights
# --------------------------------------------------------------------------------------------------
    def quantizeWeights(self):
        """ Quantizes the weights originalWeights into 8bit/16bit using min/max
        Check the range of the weights and find the next power of 2 needed to
        represent the range.
        Stores the quantized weights into qWeightsAndBias and intWeightsAndBias

        """
        numberOfWeightArray = 0
        self.weightDecBits = []
        self.intWeightsAndBias = []

        # quantize layer by layer
        for layer in self.model.layers:
            if 'activation' in layer.name:
                continue
            print('Quantisize weights of layer: ' + layer.name)

            weightsArray = layer.get_weights()

            # dense layer has 2 numpy arrays with weights, the matrix weights and the bias: [w, b]
            for weights in weightsArray:

                intBit = self.findQRangeOfWeights(weights)

                # if we have q15 we can buffer the int bits. This increases stability
                # in q7 mode this does sometimes decrease the accuracy to much
                if self.fixPointBits == 15:
                    intBit = intBit + 2

                elif (self.fixPointBits == 7) and (intBit == 0):
                    intBit = 1

                decBit = self.fixPointBits - intBit
                self.weightDecBits.append(decBit)

                # convert to q format using same method as ARM [1]
                intWeights = np.round(weights * 2**decBit)

                # saturate range inside q7 or q15 to prevent sign cast in C code
                intWeights[intWeights > 2**self.fixPointBits -1] = 2**self.fixPointBits - 1
                intWeights[intWeights < -2**self.fixPointBits] = - 2**self.fixPointBits

                # store int16 or int8 values
                self.intWeightsAndBias.append(intWeights)

                print('WeightsArrayNr_'+str(numberOfWeightArray) +
                      ': Q' + str(self.fixPointBits - self.weightDecBits[numberOfWeightArray]) +
                      '.' + str(self.weightDecBits[numberOfWeightArray]))
                numberOfWeightArray = numberOfWeightArray + 1

    def findQRangeOfWeights(self, weights):
        """ Finds the Q Range which does not clip the weights

        weights: float values of which the ideal Q range is searched
        returns: number of integer bits required to represent "weights"
        """
        qRange = [None]*(self.fixPointBits + 1)
        for bit in range(0, self.fixPointBits + 1):
            qRange[bit] = self.findQR(bit)

        minValue = weights.min()
        maxValue = weights.max()
        highestVal = max(abs(minValue), abs(maxValue))
        for bit in range(0, self.fixPointBits + 1):
            if highestVal < qRange[bit][1]:
                # highestVal fits in the range
                # return number of integer bits required
                return bit

    def findQR(self, intBits):
        """ Calculate the range, given the number of integer bits QM.N

        intBits: number of bits used for integer part
        """
        M = intBits
        N = self.fixPointBits - M
        [minVal, maxVal] = [-2**(M - 1), (2**(M - 1) - 2**(-N))]
        return [minVal, maxVal]

# --------------------------------------------------------------------------------------------------
#                        functions to find ideal output shifts of CMSIS-NN layers
# --------------------------------------------------------------------------------------------------
    def findOutputFormat(self):
        """ Finds the optimal shift_out parameters of the dense layers.
        This is done by using fake_quant_with_min_max_vars() on the trained model.
        This functions simulates quantisation into a certain range [2].
        The range is chosen such that the accuracy is the best.
        Because we have a finite small possible ranges
        we can just try every range and see how the model evaluates.

        """
        # set start (min number of M bits in Q-format: QM.N) and
        # stop (max number of M bits in Q-format: QM.N) bits
        startBit = 0
        if self.fixPointBits == 15:
            stopBit = 8
        else:
            stopBit = 4

        # init variables
        tmpModels = [None] * 2
        weights = self.model.get_weights()
        evalResult = np.zeros([int(stopBit)])

        # get information about the dense layers
        denseLayers = self.getDenseLayersInformation()
        self.bestIntBits = np.zeros([len(denseLayers)])

        # rebuild the model into keras.engine.training.Model this allows simple
        # adding of lambda layers between layers
        tmpModels[0] = self.buildNewModel(self.model)

        # find the best Q range output for each dense layer
        for denseLayer in range(0, len(denseLayers)):
            print('Calculating output format of layer: ' + str(denseLayer))
            for intBit in range(startBit, stopBit):

                # add a empty lambda layer after denseLayer
                tmpModels[1] = self.buildNewModelWithOneLambda(
                    tmpModels[0], denseLayers, denseLayer)

                # build the model with best output range lambda layers
                self.bestIntBits[denseLayer] = intBit
                tmpModels[1] = self.buildNewModelWithAllLambda(
                    tmpModels[1], self.bestIntBits, weights)

                # evaluate accuracy of the model depending on intBit and store result
                evalResult[intBit] = tmpModels[1].evaluate(self.evalData[0],
                                                           self.evalData[1], verbose=0)[1]

            # store the best output range/ interger bit which has the highest accuracy
            self.bestIntBits[denseLayer] = evalResult.argmax()
            tmpModels[0] = tmpModels[1]

        # print the output range
        for denseLayer in range(0, len(denseLayers)):
            print('Dense_'+str(denseLayer) + ' Output range: Q'
                  + str(int(self.bestIntBits[denseLayer])) + '.'
                  + str(int(self.fixPointBits - self.bestIntBits[denseLayer])))

    def buildNewModel(self, model):
        """ Rebuilds model into type which we can insert lambda layers

        model:   keras.sequential model
        returns: keras.engine.training.Model
        """
        inputs = Input(shape=(model.input_shape[1],))
        layerBefore = inputs
        # parse the model
        for layer in model.layers:
            nextLayer = self.getLayer(layer)(layerBefore)
            layerBefore = nextLayer
        output = nextLayer
        return Model(inputs, output)

    def getLayer(self, layer):
        """ Returns a Layer depending on the input can parse activation and dense layers
        """
        if "dense" in layer.name:
            return Dense(layer.output_shape[1], name=layer.name)
        if "activation" in layer.name:
            return Activation(layer.get_config().get('activation'), name=layer.name)
        if "Lambda" in layer.name:
            return Lambda(self.quantLayer,
                          output_shape=self.quantLayerShape,
                          arguments={'clipRange': self.findQR(
                              0), 'numBits': self.fixPointBits},
                          name=layer.name)

    def buildNewModelWithAllLambda(self, model, intBits, weights):
        """ Rebuilds model and sets the arguments of the fake_quant lambda layers

        model:       keras.sequential model
        intBits:     number of integer part for the dense layers
        weights:     weights to set for the new model
        returns:     keras.sequential model with fixed lambda layers which should have
                     the ideal 8/16 bit output range
        """

        idx = 0
        inputs = Input(shape=(model.input_shape[1],))
        layerBefore = inputs

        # rebuild the model and set the parameters of the fake_quant lambda layer
        for layer in model.layers:
            if "input" in layer.name:
                continue
            currentLayer = self.getLayer(layer)
            if "Lambda" in currentLayer.name:
                Lname = 'fixLambda_' + str(idx)
                currentLambda = Lambda(self.quantLayer,
                                       output_shape=self.quantLayerShape,
                                       arguments={'clipRange': self.findQR(intBits[idx]),
                                                  'numBits': self.fixPointBits},
                                       name=Lname)
                nextLayer = currentLambda(nextLayer)
                idx = idx + 1
            else:
                nextLayer = currentLayer(layerBefore)
            layerBefore = nextLayer
            output = nextLayer

        # compile and prepare the new model for evaluation
        model = Model(inputs, output)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # set the weights of model
        model.set_weights(weights)
        return model

    def buildNewModelWithOneLambda(self, model, denseLayers, denseLayer):
        """ Rebuilds model and adds a empty lambda layer after denseLayer

        model:        keras.sequential model
        denseLayers:  all dense layers of the model
        denseLayer:   after this dense layer a lambda layer will be added
        returns:      keras.sequential model with one additional lambda layer
        """
        inputs = Input(shape=(model.input_shape[1],))
        layerBefore = inputs
        for layer in model.layers:
            if "input" in layer.name:
                continue
            currentLayer = self.getLayer(layer)
            if currentLayer.name == denseLayers[denseLayer].name:
                nextLayer = currentLayer(layerBefore)
                # arguments value has to be set after!
                currentLambda = Lambda(self.quantLayer,
                                       output_shape=self.quantLayerShape,
                                       arguments={'clipRange': [
                                           0, 0], 'numBits': self.fixPointBits},
                                       name='curLambda')
                nextLayer = currentLambda(nextLayer)
            else:
                nextLayer = currentLayer(layerBefore)
            layerBefore = nextLayer
            output = nextLayer

        # compile and prepare the new model for evaluation
        return Model(inputs, output)

    def getDenseLayersInformation(self):
        """ Returns a Layer depending on the input
        can parse activation and dense layers
        """
        denseLayers = []
        for layer in self.model.layers:
            if "dense" in layer.name:
                denseLayers.append(
                    Dense(layer.output_shape[1], name=layer.name))
        return denseLayers

    @staticmethod
    def quantLayer(x, clipRange, numBits):
        """ Function of the lambda layer used to simulate quantisation to numBits
        between the layers
        -> use real shift does not really work because we simply do best
        if we just do not shift because tensorflow does not use fixpoint arithmetic
        inside the layers opposite to ARM
        # x = tf.to_int32(x)
        # x = tf.bitwise.right_shift(x, outShift)
        # x = tf.to_float(x)
        """
        x = tf.fake_quant_with_min_max_vars(x, min=clipRange[0], max=clipRange[1],
                                            num_bits=numBits)
        return x

    @staticmethod
    def quantLayerShape(inputShape):
        """ Dummy function required for the lambda layer
        """
        return inputShape

# --------------------------------------------------------------------------------------------------
#                                     functions to write C files
# --------------------------------------------------------------------------------------------------
    def storeWeights(self):
        """ Store the weights and biases
        """
        idx = 1
        fp = self.outputFilePath + 'Inc/weights.h'
        print('Write weights and net parameters into:' + fp)
        with open(fp, 'w') as f:
            f.write('#include "arm_nnfunctions.h"\n')
            for i in range(0, len(self.intWeightsAndBias), 2):
                # weights of dense layer
                if self.fixPointBits == 7:
                    f.write('q7_t aq7_layer_' + str(idx) + '_weights[] = {')
                elif self.fixPointBits == 15:
                    f.write('q15_t aq15_layer_' + str(idx) + '_weights[] = {')
                np.transpose(self.intWeightsAndBias[i]).tofile(
                    f, sep=", ", format="%d")
                f.write('};\n')

                # bias of dense layer
                if self.fixPointBits == 7:
                    f.write('q7_t aq7_layer_' + str(idx) + '_bias[] = {')
                elif self.fixPointBits == 15:
                    f.write('q15_t aq15_layer_' + str(idx) + '_bias[] = {')
                np.transpose(
                    self.intWeightsAndBias[i + 1]).tofile(f, sep=", ", format="%d")
                f.write('};\n')
                idx = idx + 1

    def storeDimension(self):
        """ Store the input and output dimensions of each layer
        """
        idx = 1
        fp = self.outputFilePath + 'Inc/weights.h'
        with open(fp, 'a') as f:
            for i in range(0, len(self.intWeightsAndBias), 2):
                f.write('#define LAYER_' + str(idx) + '_IN_DIM '
                        + str(self.intWeightsAndBias[i].shape[0]) + '\n')
                f.write('#define LAYER_' + str(idx) + '_OU_DIM '
                        + str(self.intWeightsAndBias[i + 1].shape[0]) + '\n')
                idx = idx + 1

    def storeOutShiftParams(self):
        """ Calculate and store the input and output shift

        Shift parameters are calculated as following:
        bias_shift = inputFracBits + weightsFracBits - biasFracBits
        out_shift = inputFracBits + weightsFracBits - fixPointBits - bestIntBits

        For example:
        If the input is Q5.2, the weight is Q1.6, the bias is Q4.3
        and the output is Q3.4, the product of input and weight is Q23.8
        In this case, bias_shift is 5 (left shift Q4.3 bias by 5 bits to match Q23.8)
        and out_shift is 4 (i.e., right shift Q23.8 by 4 to match
        Q3.4 output) [3].
        """
        fp = self.outputFilePath + 'Inc/weights.h'
        with open(fp, 'a') as f:
            layerInputDecBits = self.fixPointBits
            inputShift = layerInputDecBits + \
                self.weightDecBits[0] - self.weightDecBits[1]
            outputShift = layerInputDecBits + \
                self.weightDecBits[0] - \
                (self.fixPointBits - self.bestIntBits[0])
            f.write('#define LAYER_1_IN_SHIFT ' + str(int(inputShift)) + '\n')
            f.write('#define LAYER_1_OU_SHIFT ' + str(int(outputShift)) + '\n')
            # store the rest of the shift parameters
            idx = 2
            # first layer shifts is written only numberOfDenselayers - 1 to do
            for i in range(0, self.numberOfDenselayers - 1):
                layerInputDecBits = self.fixPointBits - self.bestIntBits[0]
                inShift = layerInputDecBits + \
                    self.weightDecBits[idx] - self.weightDecBits[idx + 1]
                if inShift < 0:
                    print('Warning inShift truncated!!')
                    inShift = 0
                outShift = layerInputDecBits + \
                    self.weightDecBits[idx] - \
                    (self.fixPointBits - self.bestIntBits[i + 1])
                if outShift < 0:
                    print('Warning outShift truncated!!')
                    outShift = 0
                f.write('#define LAYER_' + str(i + 2) + '_IN_SHIFT ' + str(int(inShift)) + '\n')
                f.write('#define LAYER_' + str(i + 2) + '_OU_SHIFT ' + str(int(outShift)) + '\n')
                idx = idx + 2

    def storeBitSize(self):
        """ Write header file with the FIXPOINT_MODE define such that the c compiler
        uses the correct functions.
        """
        fp = self.outputFilePath + 'Inc/bit_size.h'
        print('Write FIXPOINT_MODE_ define into:' + fp)
        with open(fp, 'w') as f:
            # add a define depending on used function (q7 or q15)
            f.write('#define FIXPOINT_MODE_' + str(self.fixPointBits) + '\n')

    def storeNetFunction(self):
        """ Creates a c function which calls the required arm functions given a net
        Restrictions:
            only dense layers (fully connected)
            only relu, softmax, tanh, sigmoid activations
            softmax can only be the last activation function

        mc: classifier with a keras.engine.sequential.Sequential
        """
        currDense = 1
        fp = self.outputFilePath + 'Inc/fully_connected.h'
        print('Write fully_connected header into:' + fp)

        # header file
        with open(fp, 'w') as f:

            # doxy description for module
            self.writeHeader(f, False)

            # inluce files
            self.writeInclude(f, False)

            # public function
            if self.fixPointBits == 7:
                f.write('uint8_t fully_connected_run(q7_t * q7_input_data);\n')
            elif self.fixPointBits == 15:
                f.write('uint8_t fully_connected_run(q15_t * q15_input_data);\n')

        # fully connected module
        fp = self.outputFilePath + 'Src/fully_connected.c'
        print('Write fully_connected source into:' + fp)
        with open(fp, 'w') as f:

            # doxy description for module
            self.writeHeader(f, True)

            # inluce files
            self.writeInclude(f, True)

            # find layer with the largest output shape layer
            currHighest = 0
            for layer in self.model.layers[0:len(self.model.layers)]:
                if "dense" in layer.name:
                    if layer.output_shape[1] > currHighest:
                        currHighest = layer.output_shape[1]

            # create buffer, LAYER_1_IN_DIM is allways 756 with MNIST
            if self.fixPointBits == 7:
                f.write(
                    'q15_t aq15_in_Buf[LAYER_1_IN_DIM];\n')
                f.write('q7_t aq7_out_Buf[' + str(currHighest) + '];\n\n')
                f.write('uint8_t fully_connected_run(q7_t * aq7_input_data)\n{\n')
            elif self.fixPointBits == 15:
                f.write('q15_t aq15_out_Buf[' + str(currHighest) + '];\n\n')
                f.write('uint8_t fully_connected_run(q15_t * aq15_input_data)\n{\n')

            # local variables
            f.write('    int16_t i16_max_val = 0x8000, i = 0;\n')
            f.write('    uint8_t u8_max_prediction = 0;\n\n')

            # add the first layer, it is different because it takes
            # aq7_input_data as input
            self.writeFirstLayer(f)
            currDense = currDense + 1

            # write the rest of the net
            for layer in self.model.layers[1:len(self.model.layers)]:
                if "dense" in layer.name:
                    # Add a fully connected function call
                    self.writeLayer(f, currDense)
                    currDense = currDense + 1
                elif "activation" in layer.name:
                    # Add a activation function call
                    act = layer.get_config().get('activation')
                    self.writeActivation(f, act, currDense)

            # get the prediction (highest value of output buffer)
            f.write('    for(i = 0; i < LAYER_' +
                    str(currDense - 1) + '_OU_DIM; i++) {\n')
            if self.fixPointBits == 7:
                f.write('        if(i16_max_val < aq7_out_Buf[i]) {\n')
                f.write('            i16_max_val = aq7_out_Buf[i];\n')
            elif self.fixPointBits == 15:
                f.write('        if(i16_max_val < aq15_out_Buf[i]) {\n')
                f.write('            i16_max_val = aq15_out_Buf[i];\n')
            f.write('            u8_max_prediction = i;\n')
            f.write('        }\n')
            f.write('    }\n')
            f.write("    return u8_max_prediction;\n")
            # end the function
            f.write('}\n')

    def writeHeader(self, f, sFile):
        """ Write module description to f

        f: file pointer
        sFile: True if f is the .c file
        """
        now = datetime.datetime.now()
        if sFile:
            f.write("/** \n")
            f.write(" * @brief       Autogenerated module by Keras2arm.py\n")
            f.write(" *              This module contains a function which can be used to\n")
            f.write(" *              classify MNIST images using the ARM-CMSIS-NN lib\n")
            if self.fixPointBits == 7:
                f.write(" * @Note        This module uses the q7 implementation\n")
            elif self.fixPointBits == 15:
                f.write(" * @Note        This module uses the q15 implementation\n")
            f.write(" * @date        " + now.strftime("%Y-%m-%d %H:%M") + '\n')
            f.write(" * @file        fully_connected.c\n")
            f.write(" * @author      Raphael Zingg zing@zhaw.ch\n")
            f.write(" * @copyright   2018 ZHAW / Institute of Embedded Systems\n")
            f.write(" */\n")
        else:
            f.write("/**\n")
            f.write(
                " * @brief       Autogenerated header file of the fully_connected module\n")
            if self.fixPointBits == 7:
                f.write(" * @Note        This module uses the q7 implementation\n")
            elif self.fixPointBits == 15:
                f.write(" * @Note        This module uses the q15 implementation\n")
            f.write(" * @date        " + now.strftime("%Y-%m-%d %H:%M") + '\n')
            f.write(" * @file        fully_connected.h\n")
            f.write(" * @author      Raphael Zingg zing@zhaw.ch\n")
            f.write(" * @copyright   2018 ZHAW / Institute of Embedded Systems\n")
            f.write(" */\n")

    @staticmethod
    def writeInclude(f, sFile):
        """ Write includes for .h and .c file f

        f: file pointer
        sFile: True if f is the .c file
        """
        if sFile:
            f.write('#include "arm_nnfunctions.h"\n')
            f.write('#include "weights.h"\n')
            f.write('#include "fully_connected.h"\n')
            f.write('#include <stdint.h>\n\n')
        else:
            f.write('#include "arm_nnfunctions.h"\n')

    def writeFirstLayer(self, f):
        """ Write first dense layer function

        f: file pointer
        """
        if self.fixPointBits == 7:
            f.write('    arm_fully_connected_q7(\n')
            f.write('        aq7_input_data,\n')
            f.write('        aq7_layer_1_weights,\n')
            f.write('        LAYER_1_IN_DIM,\n')
            f.write('        LAYER_1_OU_DIM,\n')
            f.write('        LAYER_1_IN_SHIFT,\n')
            f.write('        LAYER_1_OU_SHIFT,\n')
            f.write('        aq7_layer_1_bias,\n')
            f.write('        aq7_out_Buf,\n')
            f.write('        aq15_in_Buf);\n')
        elif self.fixPointBits == 15:
            f.write('    arm_fully_connected_q15(\n')
            f.write('        aq15_input_data,\n')
            f.write('        aq15_layer_1_weights,\n')
            f.write('        LAYER_1_IN_DIM,\n')
            f.write('        LAYER_1_OU_DIM,\n')
            f.write('        LAYER_1_IN_SHIFT,\n')
            f.write('        LAYER_1_OU_SHIFT,\n')
            f.write('        aq15_layer_1_bias,\n')
            f.write('        aq15_out_Buf,\n')
            f.write('        NULL);\n')

    def writeLayer(self, f, nd):
        """ Write functions call for dense layers

        f:   file pointer
        nd:  number of the dense layers already parsed
        """
        if self.fixPointBits == 7:
            f.write('    arm_fully_connected_q7(\n')
            f.write('        aq7_out_Buf,\n')
            f.write('        aq7_layer_' + str(nd) + '_weights,\n')
            f.write('        LAYER_' + str(nd) + '_IN_DIM,\n')
            f.write('        LAYER_' + str(nd) + '_OU_DIM,\n')
            f.write('        LAYER_' + str(nd) + '_IN_SHIFT,\n')
            f.write('        LAYER_' + str(nd) + '_OU_SHIFT,\n')
            f.write('        aq7_layer_' + str(nd) + '_bias,\n')
            f.write('        aq7_out_Buf,\n')
            f.write('        aq15_in_Buf);\n')
        elif self.fixPointBits == 15:
            f.write('    memcpy(aq15_input_data, aq15_out_Buf, sizeof(aq15_out_Buf));\n')
            f.write('    arm_fully_connected_q15(\n')
            f.write('        aq15_input_data,\n')
            f.write('        aq15_layer_' + str(nd) + '_weights,\n')
            f.write('        LAYER_' + str(nd) + '_IN_DIM,\n')
            f.write('        LAYER_' + str(nd) + '_OU_DIM,\n')
            f.write('        LAYER_' + str(nd) + '_IN_SHIFT,\n')
            f.write('        LAYER_' + str(nd) + '_OU_SHIFT,\n')
            f.write('        aq15_layer_' + str(nd) + '_bias,\n')
            f.write('        aq15_out_Buf,\n')
            f.write('        NULL);\n')

    def writeActivation(self, f, act, nd):
        """ Write functions call for activation layer

        f:    file pointer
        act:  string of activation function, ['tanh', 'relu', 'sigmoid', 'softmax']
        nd:   number of the dense layers already parsed
        """
        # TANH or SIGMOID
        if act == 'tanh' or act == 'sigmoid':
            if self.fixPointBits == 7:
                f.write('    arm_nn_activations_direct_q7(\n')
                f.write('        aq7_out_Buf,\n')
            elif self.fixPointBits == 15:
                f.write('    arm_nn_activations_direct_q15(\n')
                f.write('        aq15_out_Buf,\n')
            f.write('        LAYER_' + str(nd - 1) + '_OU_DIM,\n')
            # optimal integer part is: 7 - the shiftParameter
            # but it has to be <= 3 see source code of arm_nn_activations_direct_q7:
            # https://arm-software.github.io/CMSIS_5/NN/html/group__Acti.html
            intRange = self.bestIntBits[nd - 2]
            if intRange > 3:
                print("Warning: Saturation in Activation of dense layer:" + str(nd))
                intRange = 3
            f.write('        ' + str(int(intRange)) + ',\n')
            if act == 'tanh':
                f.write('        ARM_TANH);\n')
            elif act == 'sigmoid':
                f.write('        ARM_SIGMOID);\n')

        # RELU
        elif act == 'relu':
            if self.fixPointBits == 7:
                f.write('    arm_relu_q7(\n')
                f.write('        aq7_out_Buf,\n')
            elif self.fixPointBits == 15:
                f.write('    arm_relu_q15(\n')
                f.write('        aq15_out_Buf,\n')
            f.write('        LAYER_' + str(nd - 1) + '_OU_DIM);\n')

        # SOFTMAX
        elif act == 'softmax':
            if self.fixPointBits == 7:
                f.write('    arm_softmax_q7(\n')
                f.write('        aq7_out_Buf,\n')
                f.write('        LAYER_' + str(nd - 1) + '_OU_DIM,\n')
                f.write('        aq7_out_Buf);\n')
            elif self.fixPointBits == 15:
                f.write('    memcpy(aq15_input_data, aq15_out_Buf, sizeof(aq15_out_Buf));\n')
                f.write('    arm_softmax_q15(\n')
                f.write('        aq15_input_data,\n')
                f.write('        LAYER_' + str(nd - 1) + '_OU_DIM,\n')
                f.write('        aq15_out_Buf);\n')
