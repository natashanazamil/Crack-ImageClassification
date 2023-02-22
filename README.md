# Crack-ImageClassification

Concrete cracks are a threat to the safety and durability of buildings when it is not treated. However, some may tend to overlook and it requires manpower and time to identify each and every one of them. Hence by having it classified through images is a much efficient solution. 

With an F1 Score of 100%, this model may aid in ensuring the our safety. 

## Data Loading
Data Source:
https://data.mendeley.com/datasets/5y9wdsg2zt/2

## Data Preprocessing
The dataset contains 40000 images which are seperated into their class names. It contains 2 classes, 'Negative' and 'Positive'. They are then seperated into training, validation and testing datasets as per below:
* Training: 80%
* Validation: 10%
* Testing: 10%
 They are then converted into prefetch datasets using the `.prefetch()` method
 
## Data Augmentation
The data is then augmented using tensorflows
* `layers.RandomFlip('horizontal')`
* `layers.RandomRotation('0.2')`

<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/data_aug.PNG" alt="Augmented Data Sample">
  <br>
  <em>Augmented Data Sample</em>
</p>

## Transfer Learning
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/eval_bfr_training.PNG" alt="Evaluation Before Training">
  <br>
  <em>Evaluation Before Training</em>
</p>

<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/tensorboard_acc.PNG" alt="Tensorboard - Epoch Accuracy">
  <br>
  <em>Tensorboard - Epoch Accuracy</em>
</p>
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/tensorboard_loss.PNG" alt="Tensorboard - Epoch Loss">
  <br>
  <em>Tensorboard - Epoch Loss</em>
</p>

## Model Testing
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/eval_after_training.PNG" alt="Evaluation After Training">
  <br>
  <em>Evaluation After Training</em>
</p>
## Model Deployment

## Model Analysis
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/confusion_matrix.PNG" alt="Confusion Matrix">
  <br>
  <em>Confusion Matrix</em>
</p>
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/classification_report.PNG" alt="Classification Report">
  <br>
  <em>Classification Report</em>
</p>
<p align="center">
  <img src="https://github.com/natashanazamil/Crack-ImageClassification/blob/main/images/predicted_samples.PNG" alt="Predicted Samples">
  <br>
  <em>Predicted Samples</em>
</p>
