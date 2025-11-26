# Tuberculosis Detection from Chest X-Rays using Deep Learning
## 📖 Project Overview
This project focuses on building a deep learning model to automatically classify chest X-ray images into two categories: Healthy and Tuberculosis (TB) Positive. The goal is to assist in the early and accurate detection of Tuberculosis, a major global health challenge, by leveraging computer vision and convolutional neural networks (CNNs).

The model is built using TensorFlow/Keras and utilizes a MobileNetV2 architecture, fine-tuned on a specialized dataset of chest X-rays.

## 🗂️ Dataset
The dataset is organized into training and test sets, each containing two subdirectories:

1. healthy: Chest X-ray images of healthy patients.

2. tb: Chest X-ray images of patients with Tuberculosis.

### Dataset Summary:

* Training Images (Healthy): 152

* Training Images (TB): 151

* Testing Images (Healthy): 50

* Testing Images (TB): 50

The dataset is well-balanced, providing a solid foundation for training a reliable classification model.

## 🔎 Exploratory Data Analysis (EDA)
Initial analysis confirmed the image characteristics:

* Image Format: The X-ray images are primarily Grayscale.

* Image Dimensions: All images were resized or found to be (224, 224) pixels.

* Pixel Distribution: Histograms were used to visualize the pixel intensity distribution of sample images to understand their range and overall characteristics.

# Methodology
## Data Preprocessing
1. Images are resized to (224, 224) pixels.

2. Pixel values are normalized to the range [0, 1].

Data Augmentation is applied to the training set to improve model generalization, including:

* Rotation

* Zoom

* Width Shifting
## Models Used

### 🔹 **Model 1: Custom CNN**
A baseline CNN architecture is trained from scratch to establish benchmark performance.

### 🔸 **Model 2: MobileNetV2 (Transfer Learning)**
MobileNetV2 is fine-tuned on the same dataset to leverage pre-trained weights for improved accuracy and robustness.


## Training Configuration
* Optimizer: Adam.

* Callbacks: Early Stopping (to prevent overfitting) and ReduceLROnPlateau (to dynamically adjust the learning rate) were implemented.

* Epochs: Both models were trained for a maximum of 50 epochs.



## 🛠️ Tools & Libraries
- Python, TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-Learn

## 📈 Model Results

| Metric     | CNN Model | MobileNetV2 |
|------------|-----------|-------------|
| Accuracy   | *0.6000* | *0.84* |
| Precision  | *0.6667* | *0.8684* |
| Recall     | *0.7000* | *0.8000* |

## Evaluation
### Confusion Matrix Visualization

The matrix below details the 100 test predictions, highlighting the model's accuracy in distinguishing between the two classes:

| Metric     | CNN Model | MobileNetV2 |
|------------|-----------|-------------|
| Accuracy   | *0.6000* | *0.84* |
| Precision  | *0.6667* | *0.8684* |
| Recall     | *0.7000* | *0.8000* | 

## 🔍 Interpretation of Comparative Results
MobileNetV2 Strengths (vs. Custom CNN):
* Diagnosis Success (TP): Identified 5 more true TB cases ($\mathbf{40}$ TP vs. $35$ TP).
* Missed Cases (FN): Resulted in significantly fewer missed TB cases ($\mathbf{10}$ FN vs. $15$ FN), directly leading to its higher Recall ($0.80$).
* Healthy Identification (TN): Correctly identified nearly twice as many healthy patients ($\mathbf{44}$ TN vs. $25$ TN).
* False Alarms (FP): Generated far fewer false alarms ($\mathbf{6}$ FP vs. $25$ FP), demonstrating superior Specificity.

## Detailed Performance Metrics
The model demonstrates strong, balanced performance, with high Sensitivity being the most important factor in this medical context.


|Metric |Custom CNN|	MobileNetV2|
|---------------------------|----------|-------------|
|Sensitivity(Recall)|	*0.4200*|	*0.8400*|
|Specificity|	*1.0000*|	*0.9000*|
|PPV(Precision)|	*1.000*|	*0.8936*|
|NPV|	*0.6329*|	*0.8491*|

## 📈 ROC Curve Analysis
The Area Under the Curve (AUC) is a crucial measure of the model's ability to discriminate between classes across all possible thresholds.
* MobileNetV2 (AUC = $\mathbf{0.93}$): Delivers excellent diagnostic performance, confirming high sensitivity and specificity across all thresholds.
* Custom CNN (AUC = $\mathbf{0.87}$): Performs well but falls significantly below MobileNetV2, though it still shows solid discrimination ability.

![ROC GRAPH](ROC graph/Screenshot (1129).png)

## Conclusion
The MobileNetV2 model significantly outperformed the Custom CNN in overall accuracy, precision, and recall on the test set. Its high recall is particularly valuable for this medical application.

## Recommendations
1. Based on the project's results, the following steps are recommended for future improvement:

2. Adopt MobileNetV2 as the Primary Model: Given its superior performance, MobileNetV2 should be the focus for production and further development.

3. Increase Dataset Size and Diversity: The current dataset is relatively small. Expanding the dataset would likely improve the model's generalization capabilities and reduce the risk of overfitting.

4. Explore Advanced Augmentation Techniques: Beyond basic augmentation, techniques like CutMix, MixUp, or elastic deformations can further diversify the training data and enhance model robustness.
