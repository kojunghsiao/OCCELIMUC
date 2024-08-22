# Title: Ulcerative Colitis Severity Classification using EfficientNetB4 Transfer Learning and Ordinal Categorical Cross-Entropy Loss Function
## Abstract: Ulcerative colitis (UC) is a prevalent inflammatory bowel disease requiring accurate severity assessment for optimal treatment. This study aims to develop a deep learning-based framework for automated UC severity classification using the LIMUC dataset. To address the limitations of traditional cross-entropy loss functions, we introduced a novel Ordinal Categorical Cross-Entropy (OCCE) loss tailored for ordinal classification tasks. Our framework integrates EfficientNetB4 Transfer Learning with advanced data augmentation techniques, such as elastic transformations, and incorporates squeeze-and-excitation (SE) and spatial attention layers to enhance feature extraction and classification accuracy. For model interpretability, Grad-CAM and SmoothGrad were employed to visualize the specific regions influencing the predictions. Our model outperformed the existing benchmarks and significantly improved the accuracy and consistency of UC severity classification. This framework provides valuable insights for gastroenterologists, leading to better-informed treatment decisions and improved patient outcomes. In addition, our findings suggest potential applications in other medical image analysis tasks, offering broader diagnostic benefits.

## Schematic of the proposed model
<img width="451" alt="image" src="https://github.com/user-attachments/assets/03719b88-e95c-468a-8752-f3b29a43badf">



## Before running the following codes, it is necessary to install specific version of tensorflow
### For saving and loading the best model to and from training with h5 format, it is necessary to install tensorflow 2.15.0, keras 2.15.0, or it is not able to generate the expected results.

## 1. Training.ipynb
## Owing to the limitation resource in running on Colab, we set the 10 fold validation manully. Therefore, it is necessary to set, as the 7th fold example below:
### 1.1 train_index, val_index = folds[6]
### 1.2 checkpoint = ModelCheckpoint('/content/drive/My Drive/LIMUC/Temp/RBSLF_best_model_fold_7.h5', monitor='val_loss', save_best_only=True, mode='min')
### 1.3 load the best weights and evaluate the model on the test set. model.load_weights('/content/drive/My Drive/LIMUC/Temp/RBSLF_best_model_fold_7.h5')


## 2. Classification.ipynb
### This code is to come out the Model performance evaluation and confusion matrix of all classes and Remission vs No Remission.
### In the code, it is necessary to make sure the paths to the test dataset and the saved model.
### Following is the resukts of the confusion matrix for all classes and remission vs no remission
![Figure 2 Confusion matrix for all classes](https://github.com/user-attachments/assets/43dab9b6-e6d8-4fb4-84cf-026e68c28c9c)
![Figure 2 Confusion matrix for remission vs no remission](https://github.com/user-attachments/assets/a8e26a5f-8ccf-46a3-8746-4856f3f5f3ce)

## 3. Visualizattion.ipynb
### It is visualized by combining Grad-CAM and SmoothGrad to come out heatmap to identify where is the location that the model learned and giving its related probability to each class.
### It is necessary to check the path to the datasets, load the best model saved, and identify the image path.
![Figure 3 visualization of the model's focus area for UC severity assement](https://github.com/user-attachments/assets/71efb933-0beb-4545-8e4c-1fdce519dd0a)
![Figure 4 Histogram of the prediction with classified probabilities](https://github.com/user-attachments/assets/04177a46-e405-47f7-812f-3778bf12654f)

# Data availability
## All images and dataset are collected from Zenodo (Available: https://zenodo.org/records/5827695#.ZF-92OzMJqs).
## Published online: March 14,2022 | Version 1
