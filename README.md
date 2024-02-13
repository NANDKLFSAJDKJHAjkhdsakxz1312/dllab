# Team03
- Yixuan Wu (st178402)
- Nan Jiang (st181029)

# Diabetic Retinopathy Detection
Diabetic retinopathy, a major cause of blindness linked to diabetes, is one of the major causes of blindness in the Western world. The traditional manual screening process for DR detection, however, is both expensive and time-consuming. To address these issues, we have developed deep neural network models, which are designed to streamline the DR detection process, enhancing efficiency and reliability. 

## Binary Classification task
### Configuration Setup

Before you start, you need to configure the paths for your project files and dataset to ensure the code can access the required files correctly.

#### 1. Open Configuration File
First, navigate to the `configs/config.gin` file.

#### 2. Set Dataset Name
For this project, we specifically use the IDRID dataset for the binary classification task. So please check
```gin
load.name='idrid'
```

#### 3. Set Project Folder Location
In the configuration file, you need to specify the location of your project files. For example, if your project is located at 
`/misc/home/RUS_CIP/st178402/dl-lab-23w-team03/diabetic_retinopathy`, you should set it as follows:
    
  
    load.data_dir='/misc/home/RUS_CIP/st178402/dl-lab-23w-team03/diabetic_retinopathy'
   
   
#### 4. Set Dataset Path
Next, specify the path to your IDRID dataset. If the dataset is located at `/misc/home/data/IDRID_dataset`, configure it like this:
    
    
    prepare_image_paths_and_labels.data_dir='/misc/home/data/IDRID_dataset'
    

### Training the Model

To train the model for the binary classification task of diabetic retinopathy detection, follow these steps:

#### 1. Navigate to `main.py`.

#### 2. Choose a Model
This project offers four different models. You will need to select which model you wish to use for training. We have 3 models for binary classification: simple CNN, transferlearning with renset50 and densnet121. Change the `model_name` variable to the name of the model you want to use. For example:
  ```main
  model_name = 'simple_cnn'
  ```
  Also, update the folder variable to match the model_name:
   ```main
   folder = 'simple_cnn'
   ```
#### 3. Configure Training Parameters
Next, you need to set your preferred training parameters in the `configs/config.gin` file.
Open configs/config.gin.
Set the total number of training steps, log interval, and checkpoint interval as follows:
```gin
train.Trainer.total_steps = 10000
train.Trainer.log_interval = 100
train.Trainer.ckpt_interval = 100
```
#### 4. Training
Training logs and results will be stored under `experiment/your_model_name`.
To start training, execute the command with the appropriate flags.
```main
main.py --train=true --mode=binary
```
for example
```main
flags.DEFINE_boolean("train",True, "Specify whether to train or evaluate a model.")
flags.DEFINE_string("mode", "binary", "Specify the classification mode: binary or multiclass.")
```
Finally, run `main.py` and train the model.

### Results
#### 1.Evaluation and Metrics
To start evaluation, execute the command with the appropriate flags.
```main
main.py --train=False --mode=binary
```
for example
```main
flags.DEFINE_boolean("train",False, "Specify whether to train or evaluate a model.")
flags.DEFINE_string("mode", "binary", "Specify the classification mode: binary or multiclass.")
```
For simple CNN, we get the result:
| Label | Precision | Recall | Specificity | F1-Score |
|-------|-----------|--------|-------------|----------|
| 0     | 0.846     | 0.688  | 0.891       | 0.759    |
| 1     | 0.766     | 0.891  | 0.688       | 0.824    |


<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/68bb1f13-ba14-4185-8eeb-0dd0ead538fd" width="400">

For transferlearning, we get the results:
| Model Name    | Accuracy |
|---------------|----------|
| ResNet50      | 61.17%   |
| DenseNet121   | 65.05%   |

#### 2.Deep Visualization
- Navigate to `visualization/gradcam.py`
- Place an image, such as `IDRiD_102.jpg`, in the same directory as `gradcam.py` for easy access.

- Update the grad_cam Function Parameters
Ensure the `grad_cam` function is correctly configured:
- Change `model.get_layer('your_layer_name').output` to target the specific layer whose outputs you want to visualize. Typically, this is the last convolutional layer of the model. Replace `your_layer_name` with the actual name of the layer you wish to use.

- Ensure Correct Model Path
The model path is specified as follows:
```text
experiments/your_model_id/ckpts/saved_model
```
![visual](https://media.github.tik.uni-stuttgart.de/user/7276/files/73f1c762-98ae-40e8-98c0-0ce90b89d77b)

## Multi-Class Classification task
### Configuration Setup
#### 1. Set Dataset Name
For this project, we specifically use the Kaggle Challenge Dataset provided by EyePACS dataset for the binary classification task. So please check
```gin
load.name='eyepacs'
```
#### 2. Set Dataset Path
Go to `input_pipeline/dataset.py` and find function get_eyepacs_tfrecord(), the base path is according to the location of EyePACS dataset.Like:
``` 
    base_path = '/misc/home/data/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0'
```
### Training
Like binary task, change the `model_name` and 'folder'
```main
main.py --train=true --mode=multi
```
### Results
By using simple CNN model, we get the results:
| Label | Precision | Recall | Specificity | F1-Score |
|-------|-----------|--------|-------------|----------|
| 0     | 0.636     | 0.786  | 0.338       | 0.703    |
| 1     | 0.040     | 0.094  | 0.929       | 0.056    |
| 2     | 0.220     | 0.149  | 0.853       | 0.178    |
| 3     | 0.304     | 0.092  | 0.983       | 0.141    |
| 4     | 0.458     | 0.127  | 0.987       | 0.199    |

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/dfa83e15-1f17-4cd1-8e90-7f182f012b7c" width="500">


## Regression task
For the regression task, we utilise a single model called 'simple_cnn_regression'.
### Training
Navigate directly to `main.py` and modify the `model_name` to `simple_cnn_regression` and as well as the `folder`.
### Results
For regression task, we choose mean squared error and mean absolute error as our metrics.
| Model     | Value |
|-----------|-------|
| Test MAE  | 1.99  |
| Test MSE  | 4.76  |


# Human Activity Recognition
In this task, we use 2 kinds of labeling ways: S2S and S2L. Yixuan is responsible for S2S and Nan is responsible for S2L. Thus, we have 2 kinds of codes.
## S2S
### Configuration Setup
#### 1. Open Configuration File
First, navigate to the `configs/config.gin` file.

#### 2. Set Dataset Name
For this project, we specifically use the IDRID dataset for the binary classification task. So please check
```gin
load.name='HAPT'
```

### Set Dataset Path
Go to `input_pipeline/preprocessing_visualization.py` and find `data_dir` and `labels_file` the path is according to the location of HAPT dataset.
``` 
    data_dir = '/misc/home/data/HAPT_dataset/RawData'
    labels_file = '/misc/home/data/HAPT_dataset/RawData/labels.txt'
```

### Training the Model

To train the model for human activity recognition, follow these steps:

- Navigate to `main.py`.

- Choose a Model
This project offers four different models. You will need to select which model you wish to use for training. We have 2 models for S2S: RNN model and GRU model. Change the `model_name` variable to the name of the model you want to use. For example:
  ```main
  model_name = 'rnn'
  ```
  Also, update the folder variable to match the model_name:
   ```main
   folder = 'rnn_model'
   ```
- Configure Training Parameters
Next, you need to set your preferred training parameters in the `configs/config.gin` file.
Open configs/config.gin.
Set the total number of training steps, log interval, and checkpoint interval as follows:
```gin
train.Trainer.total_steps = 10000
train.Trainer.log_interval = 100
train.Trainer.ckpt_interval = 100
```
- Training
Training logs and results will be stored under `experiment/your_model_name`.
To start training, execute the command with the appropriate flags.
```main
main.py --train=true
```
for example
```main
flags.DEFINE_boolean("train",True, "Specify whether to train or evaluate a model.")
```
Finally, run `main.py` and train the model.

### Results
- Evaluation and Metrics
To start evaluation, execute the command with the appropriate flags.
```main
main.py --train=False
```
for example
```main
flags.DEFINE_boolean("train",False, "Specify whether to train or evaluate a model.")
```
For RNN model, we get the result:
| Label               | Precision | Recall | Specificity | F1-Score |
|---------------------|-----------|--------|-------------|----------|
| WALKING             | 0.982     | 0.915  | 0.997       | 0.947    |
| WALKING_UPSTAIRS    | 0.960     | 0.962  | 0.994       | 0.961    |
| WALKING_DOWNSTAIRS  | 0.920     | 0.977  | 0.988       | 0.948    |
| SITTING             | 0.933     | 0.940  | 0.987       | 0.936    |
| STANDING            | 0.950     | 0.926  | 0.989       | 0.938    |
| LAYING              | 0.981     | 0.974  | 0.996       | 0.977    |
| STAND_TO_SIT        | 0.526     | 0.805  | 0.994       | 0.636    |
| SIT_TO_STAND        | 0.797     | 0.798  | 0.998       | 0.797    |
| SIT_TO_LIE          | 0.696     | 0.617  | 0.996       | 0.654    |
| LIE_TO_SIT          | 0.444     | 0.659  | 0.992       | 0.531    |
| STAND_TO_LIE        | 0.557     | 0.629  | 0.993       | 0.591    |
| LIE_TO_STAND        | 0.584     | 0.535  | 0.994       | 0.558    |
<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/c71be3fe-b270-4d46-84a4-9994d9da74f6" width="700">


For GRU model, we get the result:
| Label               | Precision | Recall | Specificity | F1-Score |
|---------------------|-----------|--------|-------------|----------|
| WALKING             | 0.969     | 0.951  | 0.995       | 0.960    |
| WALKING_UPSTAIRS    | 0.969     | 0.932  | 0.995       | 0.950    |
| WALKING_DOWNSTAIRS  | 0.930     | 0.979  | 0.990       | 0.954    |
| SITTING             | 0.798     | 0.972  | 0.961       | 0.876    |
| STANDING            | 0.992     | 0.820  | 0.998       | 0.898    |
| LAYING              | 0.969     | 0.976  | 0.993       | 0.972    |
| STAND_TO_SIT        | 0.567     | 0.883  | 0.995       | 0.691    |
| SIT_TO_STAND        | 0.694     | 0.807  | 0.997       | 0.746    |
| SIT_TO_LIE          | 0.710     | 0.701  | 0.996       | 0.705    |
| LIE_TO_SIT          | 0.713     | 0.559  | 0.996       | 0.627    |
| STAND_TO_LIE        | 0.642     | 0.752  | 0.994       | 0.693    |
| LIE_TO_STAND        | 0.363     | 0.483  | 0.991       | 0.414    |

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/8ff775c4-14ea-43b5-8cd5-c44e0c7ba03f" width="700">


- Visualization
RNN model：

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/49a8f75c-661f-4708-8193-44d9719ca277" width="800">

GRU model：

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/52f6750d-42df-417f-b9b0-f55b7a839231" width="800">






## S2L
In sequence to label method, we create a label for each window, then we start to train and evaluate our model. 

### Configuration Setup

Before you start, you need to configure the paths for your project files and dataset to ensure the code can access the required files correctly.

#### 1. Open Configuration File
First, navigate to the `configs/config.gin` file.

#### 2. Set Dataset Name
For this project, we specifically use the IDRID dataset for the binary classification task. So please check
```gin
load.name='HAPT'
```

#### 3. Set Project Folder Location
In the configuration file, you need to specify the location of your project files. For example, if your project is located at 
'D:\HAR\pythonProject\input_pipeline_s2l' , you should set it as follows:
    
  
    load.data_dir=r'D:\HAR\pythonProject\input_pipeline_s2l'
    
#### 4. Set Dataset Path
The dataset is located at r'/home/data/HAPT_dataset/RawData' on the server, so set preprocessor.file_path = r'/home/data/HAPT_dataset/RawData' .

### Training
Go to main.py, set the Flag to be True, and type your model_name and folder, there are two options, namely, 'lstm' or 'crnn', then train the selected model.

### Evaluation
Go to main.py, set the flag to be False, and type your model_name and folder, same as above, then it produces the result.

### visualization
Go to visualization.py and run it. You will get the visualization for a sequence for ground truth and prediction. 

### optimization
Go to optimization.py and run it to do tuning.
