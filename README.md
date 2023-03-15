# Dělen: Enabling Flexible and Adaptive Model-serving

Dělen is  a flexible and adaptive model-serving system for multi-tenant edge AI.


## Introduction
Dělen is a flexible, adaptive, and multi-tenant model-serving system for supporting low-latency IoT applications on edge AI platforms.
Dělen provides the following key characteristics:

1. Dělen is multi-tenant in its support for multiple concurrent applications.
2. Dělen flexible in enabling applications to specify their own latency, energy, or accuracy requirements.
3. Dělen is adaptive in enabling applications to specify policies that adapt their operation under workload variations and energy constraints. 

In particular, unlike cloud model serving systems, Dělen implements model serving using multi-exit DNNs rather than model selection, since they require much less memory and permit granular control over inference requests using conditional run-time execution based on application-specific latency, energy, or accuracy constraints.

## Paper

```
@inproceedings{iotdi2023delen,
    author = {Qianlin Liang and Walid A. Hanafy and Ahmed Ali-Eldin and David Irwin  and Prashant Shenoy},
    title = {Dělen: Enabling Flexible and Adaptive Model-serving for Multi-tenant Edge AI},
    month = 5,
    year = 2023,
    booktitle = {Proceedings of IEEE/ACM Eighth International Conference on Internet-of-Things Design and Implementation (IoTDI), San Antonio},
    isbn = {979-8-4007-0037-8/23/05},
    doi = {10.1145/3576842.3582375}
}
```

## Project Structure
```
> tree .
├── LICENSE
├── README.md
├── dataset                                   # Download scripts for datasets used in the paper  
├── delen                                     # Delen source code
│   ├── conditional_executor                  # Conditional execution framework module
│   ├── examples                              # Executable examples
│   ├── models                                # Multi-exit DNNs 
│   ├── utils                                 # Common utilities
├── requirements.txt                          # Dependencies
├── resources                                 # Trained multi-exit DNN resources
│   └── resnet50                              # Multi-exit Resnet50 model            
│       ├── conditions_20ms.json              # Sample condition file1
│       ├── conditions_40ms.json              # Sample condition file2
│       ├── resnet50_profile.json             # Sample multi-exit DNN profile
│       └── trt                               # Compiled TensorRT engine
```

## Hardware Requirement
The current Dělen implementation and the compiled multi-exit models are tested only on Jetson Nano. 


## Dělen Examples

We provide a few examples to illustrate how Dělen works. These are in the `delen/examples` folder. 

### Train Multi-exit DNNs

After downloading the dataset using the download script in `dataset/`, we can train a pre-defined multi-exit DNN
using the `delen/examples/train_multi_exit_dnn.py`. The following command train a 4-exit `Resnet18`:

```
python3 train_multi_exit_dnn.py -b 128 --optimizer adam -n 20 \
                             -m resnet18 --weight 0.1 0.2 0.2 0.5 \
                             --dataset food-101 \
                             --dataset-dir ~/delen/dataset/food-101/ \
                             --input-type image \
                             -l 0.001
```

### Create Subnetworks
To enable conditional execution, we need to split the DNN model into subnetworks, one for each exit. This can be done
using the `delen/examples/create_subnetworks.py` script:

```
python3 create_subnetworks.py -m resnet18 --num-classes 101 -d tmp/train/resnet18
```
where `tmp/train/resnet18` is the model directory created by the training script. The above script will create 4 `.onnx` files,
which can be compiled to TensorRT engines. 

### Run conditional execution framework

The `delen/examples/run_condition_execution.py` provides an example to run conditional execution framework. We can run the 
script using the following script:

```
python3 run_condition_execution.py --model-dir ../../resources/resnet50/
```

The script does the following things: 

1. Load the pretrained 4-exit `Resnt50` provided in `resources/resnet50/trt`. 
2. Load the model profile provided in `resources/resnet50/resnet50_profile.json`
3. Load the exit conditions in `resources/resnet50/conditions_XXms.json`, which state that the model should exit at current exit if the response time is greater than `XX`ms. 
4. Execute the model for 10 seconds using the first condition (i.e. `response_time > 20ms`) then switch to the second condition (i.e. `response_time > 40ms`).

You should see the following output when running successfully:

```
[INFO] main task TaskState.FINISHED, response time: 26.59, exit#: 1
[INFO] main task TaskState.FINISHED, response time: 26.34, exit#: 1
[INFO] main task TaskState.FINISHED, response time: 25.68, exit#: 1
...
[INFO] main Updating conditions from file ../../resources/resnet50/conditions_40ms.json
[INFO] main task TaskState.FINISHED, response time: 43.52, exit#: 2
[INFO] main task TaskState.FINISHED, response time: 42.85, exit#: 2
[INFO] main task TaskState.FINISHED, response time: 43.25, exit#: 2
...
```

