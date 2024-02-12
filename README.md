# Team03
- Yixuan Wu (st178402)
- Nan Jiang (stXXXXXX)

# Diabetic Retinopathy Detection
Diabetic retinopathy, a major cause of blindness linked to diabetes, is one of the major causes of blindness in the Western world. The traditional manual screening process for DR detection, however, is both expensive and time-consuming. To address these issues, we have developed deep neural network models, which are designed to streamline the DR detection process, enhancing efficiency and reliability. 

# Binary Classification task
## Configuration Setup

Before you start, you need to configure the paths for your project files and dataset to ensure the code can access the required files correctly.

### 1. Open Configuration File
First, navigate to the `configs/config.gin` file.

### 2. Set Dataset Name
For this project, we specifically use the IDRID dataset for the binary classification task. So please check
```gin
load.name='idrid'
```

### 3. Set Project Folder Location
In the configuration file, you need to specify the location of your project files. For example, if your project is located at 
`/misc/home/RUS_CIP/st178402/dl-lab-23w-team03/diabetic_retinopathy`, you should set it as follows:
    ```gin
    load.data_dir='/misc/home/RUS_CIP/st178402/dl-lab-23w-team03/diabetic_retinopathy'
    ```
### 4. Set Dataset Path
Next, specify the path to your IDRID dataset. If the dataset is located at `/misc/home/data/IDRID_dataset`, configure it like this:
    ```gin
    prepare_image_paths_and_labels.data_dir='/misc/home/data/IDRID_dataset'
    ```

    Please change the above paths to reflect the actual location of your IDRID dataset.

## Training the Model

To train the model for the binary classification task of diabetic retinopathy detection, follow these steps:

### 1.Navigate to `main.py`.

### 2.Choose a Model
This project offers four different models. You will need to select which model you wish to use for training. We have 3 models for binary classification: simple CNN, transferlearning with renset50 and densnet121. Change the `model_name` variable to the name of the model you want to use. For example:
  ```main
  model_name = 'your_model_name'
  ```
  Also, update the folder variable to match the model_name:
   ```main
   folder = 'your_model_name'
   ```
### 3.Configure Training Parameters
Next, you need to set your preferred training parameters in the `configs/config.gin` file.
Open configs/config.gin.
Set the total number of training steps, log interval, and checkpoint interval as follows:
```gin
train.Trainer.total_steps = 10000
train.Trainer.log_interval = 100
train.Trainer.ckpt_interval = 100
```
### 4.Training
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

# Results
### 1.Evaluation and Metrics
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
| 0     | 0.487     | 0.655  | 0.730       | 0.559    |
| 1     | 0.844     | 0.730  | 0.655       | 0.783    |

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/a3f58349-6c5b-4299-ad21-2f150a6be4ca" width="400">

For transferlearning, we get the results:


### 2.Deep Visualization
#### Navigate to `visualization/gradcam.py`
Place an image, such as `IDRid_102.jpg`, in the same directory as `gradcam.py` for easy access.

#### Update the grad_cam Function Parameters
Ensure the `grad_cam` function is correctly configured:
- Change `model.get_layer('your_layer_name').output` to target the specific layer whose outputs you want to visualize. Typically, this is the last convolutional layer of the model. Replace `your_layer_name` with the actual name of the layer you wish to use.

#### Ensure Correct Model Path
The model path is specified as follows:
```text
experiments/your_model_id/ckpts/saved_model
```
![visual](https://media.github.tik.uni-stuttgart.de/user/7276/files/73f1c762-98ae-40e8-98c0-0ce90b89d77b)

# Multi-Class Classification task
## Configuration Setup
### 1. Set Dataset Name
For this project, we specifically use the Kaggle Challenge Dataset provided by EyePACS dataset for the binary classification task. So please check
```gin
load.name='eyepacs'
```
### 2. Set Dataset Path
Go to `input_pipeline/dataset.py` and find function get_eyepacs_tfrecord(), the base path is according to the location of EyePACS dataset.Like:
``` 
    base_path = '/misc/home/data/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0'
```
## Training
Like binary task, change the `model_name` and 'folder'
```main
main.py --train=true --mode=multi
```
## Results
By using simple CNN model, we get the results:
| Label | Precision | Recall | Specificity | F1-Score |
|-------|-----------|--------|-------------|----------|
| 0     | 0.636     | 0.786  | 0.338       | 0.703    |
| 1     | 0.040     | 0.094  | 0.929       | 0.056    |
| 2     | 0.220     | 0.149  | 0.853       | 0.178    |
| 3     | 0.304     | 0.092  | 0.983       | 0.141    |
| 4     | 0.458     | 0.127  | 0.987       | 0.199    |

<img src="https://media.github.tik.uni-stuttgart.de/user/7276/files/dfa83e15-1f17-4cd1-8e90-7f182f012b7c" width="500">


# Regression task
For the regression task, we utilise a single model called 'simple_cnn_regression'.
## Training
Navigate directly to `main.py` and modify the `model_name` to `simple_cnn_regression` and as well as the `folder`.
## Results
For regression task, we choose mean squared error and mean absolute error as our metrics.
| Model     | Value |
|-----------|-------|
| Test MAE  | 1.99  |
| Test MSE  | 4.76  |

