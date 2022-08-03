![image](https://user-images.githubusercontent.com/90834830/182668739-5815f217-f114-45a0-97e4-b9bdc906666d.png)

# Detection of Emotion of a person by training a model using CNN (Convolution Neural Network).

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.


Our proposed solution is to build a Neural Network Model which can analyze the face of a person in real time and classify the kind of emotion showed. It will use Convolutional Neural Networks and a few parameters of OpenCV along with image pre-processing. The model will be tested on internal and external webcams and will show the current emotion on display

![image](https://user-images.githubusercontent.com/90834830/182671565-d5b39f1f-23f1-4682-a940-76a9b40121d4.png)


# ALGORITHM

1.) All the relevant libraries are imported like pandas, numpy, tensorflow, 
etc.

2.) The dataset is imported and converted to a dataframe.

3.) Training and testing data is splitted.

4.) Data type of ‘Pixel’ column is converted to integer.

5.) Image is reshaped, rescales and converted to multiple images using 
rotation, flipping, etc.

6.) Train and test sets are optimized and batch size for the neural network is 
set.

7.) A CNN is formed of 3 convolution layers, 3 pooling layers, 1 flattening 
layer and finally a full connection.

8.) Output layer of dimensions 7 is formed.

9.) Model is fit into the data and accuracy is used as metric to train. Adam 
optimizer is used for Stochastic Gradient Descent.

10.) Model is saved for future use.

11.) Saved model is loaded to test in internal and external webcam.

12.) Haar cascade is used to detect face in the webcam.

13.) Model is tested and results are printed along with the detection of face 
in the dialogue box.



#  Dataset Description

• In our model we have used the fer2013 dataset which is an open-source data set that was made publicly available for a Kaggle competition.

• It contains 48 X 48-pixel grayscale images of the face. There are seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) present in the data.

• The CSV file contains two columns that are emotion that contains numeric code from 0-6 and a pixel column that includes a string surrounded in quotes for each image.

# Result
<p align="center">
<img src = "https://user-images.githubusercontent.com/90834830/182672325-9e767af7-adc7-40c4-8588-4b34c51c8178.png">

<img src = "https://user-images.githubusercontent.com/90834830/182672337-5e834887-915b-4fad-8080-c182853ed304.png">

<img src = "https://user-images.githubusercontent.com/90834830/182674014-101ee9bf-ec7e-488e-97ff-83c32cefb224.png">

<img src = "https://user-images.githubusercontent.com/90834830/182672398-8a3a3478-74b4-4523-bdbd-99ac9d829252.png">

<img src = "https://user-images.githubusercontent.com/90834830/182672418-1e3ff13e-f7d0-4887-9cc8-0c8e35bd6ac7.png">
</p>

# Conclusion

Hence successfully built a human emotion recognition model using Convolutional Neural Network. The trained model was used to categorise the emotions in real-time after we first defined the network and trained it to be capable of classifying the proper emotion. It provides good accuracy of approx. 60%. After designing a network that is capable of classifying the emotions, we use OpenCV for the detection of the faces and then pass it for prediction. 

# Team Members

<img width="949" alt="image" src="https://user-images.githubusercontent.com/90834830/182678211-a21549ed-a425-42f0-a998-5b7a1e62b2c5.png">
