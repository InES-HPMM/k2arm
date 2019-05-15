# k2arm - Forum Künstliche Intelligenz

This repository contains the [k2arm software framework](https://github.engineering.zhaw.ch/hpmm/ki_forum_stuttgart/blob/master/ARM/host/k2arm.py), related code and the presentation from the "Forum Künstliche Intelligenz" talk in Stuttgart , 14.05.2019. 

The [k2arm software framework](https://github.engineering.zhaw.ch/hpmm/ki_forum_stuttgart/blob/master/ARM/host/k2arm.py) can be used to translate a custom keras model into C-code.
The generated C-code can, in combination with the [ARM-CMSIS-NN](http://www.keil.com/pack/doc/CMSIS_Dev/NN/html/index.html) functions, 
be used to run neural-net calculations in an efficient way on an embedded microcontroller such as the Cortex-M4. Those generated networks are compared to networks which are generated with the [x-Cube-AI](https://www.st.com/en/embedded-software/x-cube-ai.html) tool from ST.

Three different sized [MNIST classifiers](https://github.engineering.zhaw.ch/hpmm/ki_forum_stuttgart/blob/master/models/) are compared, the results are included in the [presentation](https://github.engineering.zhaw.ch/hpmm/ki_forum_stuttgart/blob/master/k2arm_final.pptx).

##  Requirements and Preparation
1. Following additional software and versions is required:
   - ubnntu 18.04
   - python 3.6
   - tensorflow 1.10.0
   - keras version 2.2.4 
   - [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html)
   - [X-Cube-AI](https://www.st.com/en/embedded-software/x-cube-ai.html)
   - arm-none-eabi-gcc 6.3.1 

2. Follwoing hardware is required:
   - [STM32F4-Discorevy Board](https://www.st.com/en/evaluation-tools/stm32f4discovery.html)
   - [TLL-232R Converter](https://ch.farnell.com/ftdi/ttl-232r-3v3/kabel-usb-ttl-pegel-seriell-umsetzung/dp/1329311?mckv=s89FAqCVd_dc|pcrid|251391972450|kword|ttl-232r-3v3|match|p|plid|&CMP=KNC-GCH-GEN-SKU-MDC-German&gclid=EAIaIQobChMIjfS4hcyo2wIVxDobCh14jwVBEAAYAiAAEgLMo_D_BwE)

3. Checkout the repository
```bash
git clone https://github.com/InES-HPMM/k2arm.git
```

4. Build stlink
```bash
cd k2arm/ST/stlink
make
```

## Reproduce k2arm Measurements
1. Switch into the ARM host directory and generate the C-code with the k2arm framework
```bash
cd ../../ARM/host/
python3 generateCMSISCode.py
```
2. Switch into the target directory and build the firmware
```bash
cd ../target/
make
```
3. Connect the STM32F4-Discovery board to your computer, and flash the firmware
```bash
sudo ../../ST/stlink/build/Release/st-flash --format ihex write ./build/k2arm.hex
```
4. Switch into the top level dir
```bash
cd ../..
```
5. Open the file `eval.py` and adjust the settings serDev and modelPath
```python
serDev = '/dev/ttyUSB1'
modelPath = 'models/modelMid/model.keras'
```
6. Connect the serial device
```
PA0-WKUP Board ------> TX Serial device host
PA1      Board ------> RX Serial device host
GND      Board ------> GND host
```
7. Tun the evaluation script:
```bash
python3 eval.py
```
8. To measure the run-time of the neural net connect the GPIO `PIN0` of `GPIOB` to an oscilloscope 


## Reproduce X-Cube-AI Measurements

1. Start STM32CubeMX
2. In the section `File` chose the option to `Load Project` and select the file: `ST/St.ioc`
3. In the section `Additional Software` chose the desired network in the `network` section, then press `GENERATE CODE`
4. Go into the ST dir and checkout the wrong changed files (Makefile and application code)
```bash
cd ST/
git checkout  Src/main.c Src/app_x-cube-ai.c Inc/app_x-cube-ai.h Makefile
```
5. Build the firmware
```bash
make
```
6. Connect the STM32F4-Discovery board to your computer, and flash the firmware

```bash
sudo ./stlink/build/Release/st-flash --format ihex write ./build/ST.hex
```
7. Switch into the top level dir

```bash
cd ..
```
8. Open the file `eval.py` and adjust the settings
```python
serDev = '/dev/ttyUSB1'
modelPath = 'models/modelL/model.keras'
```
9. Connect the serial device
```
PA0-WKUP Board ------> TX Serial device host
PA1      Board ------> RX Serial device host
GND      Board ------> GND host
```
9. Run the evaluation script
```bash
python3 eval.py
```
10. To measure the run-time of the neural net connect the GPIO `PIN0` of `GPIOB` to an oscilloscope 
