# Image Classification with Convolutional Neural Networks (CNN)

## Introduction

### 1.1 Image Classification: 
Image classification is the task of assigning images to predefined classes based on their visual content. It enables many critical applications such as medical imaging, self-driving vehicles, and facial recognition systems.
<p align="center">
   <img src="https://github.com/yohanankling/Deep-Learning/assets/93263233/2d556dee-0de4-41d1-bd42-bbf247e38b22" width="450" height="300">
 </p>

### 1.2 Dataset:
The CIFAR-10 dataset used consists of 60,000 32x32 color images across 10 classes including airplanes, cars, birds, cats, deer, etc. The images have 3 channels (RGB) and integer pixel values from 0-255. The training and test sets contain 50,000 and 10,000 images respectively.

### 1.3 Method:
The dataset was split into 83.4% training, 16.6% test, and 20% validation sets. Pixel values were normalized to [0,1] by dividing by 255 to speed up learning. Labels were one-hot encoded for categorical classification. The goal is to utilize the image pixels to train the model for classifying images into one of ten predefined categories, achieving a high rate of correct classifications while avoiding overfitting.

### 1.4 Goal:
- Achieve a high rate of correct classifications while avoiding overfitting.
- Trying different models to decide which is the most suitable for the task.
- Finding the right balance between the training, validation, and test datasets in each model.

### 1.5 Load and Normalize Data:
Images are normalized to [0,1] by dividing each pixel value by 255. Labels are one-hot encoded representing each label as a vector.

## Logistic Regression Model

### 2.1 Steps:
1. **Flatten Images**: Reshape 3D images into 2D arrays.
2. **Train Model**: Train a Logistic Regression model.
3. **Predict & Evaluate**: Make predictions on unseen images and calculate accuracy.
4. **Loss**: Log loss helps to predict the probability of model accuracy and improvement potential.

### 2.2 Results:
After several improvements, the logistic model achieved an accuracy of 0.4054.

## Improve the CNN Model

### 3.1 Architecture:
The newer model has additional layers including an extra convolutional layer with 128 filters and a dense layer with 128 units.

### 3.2 Enhancements:
- **Batch Normalization**: Standardizes inputs to each layer.
- **Dropout**: Randomly drops connections during training to prevent overfitting.
- **Learning Rate Adjustment**: Exponential decay of learning rate with each epoch.

### 3.3 Results:
<p align="center">
 <img src="https://github.com/yohanankling/Deep-Learning/assets/93263233/7ab1f7ce-25d3-4498-b4cf-f780d54ce409" width="200" height="200">&nbsp;&nbsp;
  <img src="https://github.com/yohanankling/Deep-Learning/assets/93263233/be921207-96df-4cbb-b427-3c4ba62a8616" width="200" height="200">&nbsp;&nbsp;
  <img src="https://github.com/yohanankling/Deep-Learning/assets/93263233/3fd56368-c454-4dac-9a7c-9f8b68a0bf69" width="200" height="200">
  <br/> 
</p>
- With 15 epochs, an increase of 3.5% accuracy was achieved.
- The final model achieved a remarkable accuracy of 75.4%.

## Conclusion

Utilizing Convolutional Neural Networks (CNN) over logistic models has proven advantageous, resulting in improved accuracy. Incorporating architectural enhancements such as batch normalization, dropout, and dynamic learning rate adjustments significantly contributed to refining the model's performance. The CNN model achieved a notable accuracy of approximately 75% across ten distinct classes, underscoring the effectiveness of advanced techniques in developing a robust classification model.

## Acknowledgements

Special thanks to my friend [Tair Mazriv](https://github.com/TairMaz) for their valuable contributions and support throughout the development of this project. Their insights and assistance were instrumental in achieving our goals.
