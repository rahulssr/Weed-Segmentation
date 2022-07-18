### Introduction

Weed are the unwanted plants on the field, they are harmful to crop yield as they compete for the resources with
the main crop which on initial stage of the crop may lead to nutrient deficiency and ultimately crop failure. Precise weed
management is one of the difficult jobs as it includes proper seperation of weeds and crop for precise management. The
segmentation of weeds using deep learing appriach in images is of great significance for precise weeding and reducing
herbicide pollution. In this work, we are presenting an onion-crop field-specific solution for the automatic detection of
weeds in precision agriculture. In the field environment, crops and weeds are similar, so it is difficult to identify weed
and crop separately for precise weed management thus an architecture based on deep learning is proposed to segment
weeds from images followed by detection and classification. This algorithm can segment weeds from the soil as well
as from crops in images. This multiclass semantic segmentation is based upon the combination of both the binary and multi segmentation models

### Overview of the Model

<img src= images/Overview.png  width = "1000" height = "500">


### Dataset
The images was taken from majorly four angles namely
top,bottom,left and right .The images of different segments
of fields were taken consisting of onion crop and weeds
of slightly different age or growth stage so as to maintain
versatility and generality in dataset ,which would help in
better training of the model. The entire work of building and
collection of dataset was carried out with the help of our
collaborator - Indira Gandhi Krishi Vishwavidyalaya, Raipur.
It also included some practical scenarios which appears on
the fields including shadows and difference in amount of
light at different time of the day

<img src= images/dataset.jpg  width = "800" height = "400">

### Model Architecture

In a vanilla UNET there are skip connections i.e each layer
of encoder part is concatanated with the respective layers of
the decoder part illustrated in 
The Skip connections primarily helps in capturing the lowlevel features such as edges, color, gradient orientation and
then combining with the later layers so that there is no case
of feature missing Here, minute details such as onion crop
leaf texture, image characteristics, texture and color coding of
background shadows and contrast areas The loss function in
the vanilla UNET model is binary cross entropy(BCE)

<img src= images/unet.png  width = "1000" height = "500">




Our model is a combination of two UNET with certain skip connections illustrated in architecture The left part
of the model segments the image into binary segmentation
and acts as input for the right part of the model which then
multi segments the image and thus the combination of both is
formed
The loss function is BCE Loss of both the binary part and
multi part

J(y) = BCEB(y) + BCEM(y) 

<img src= images/U2.png  width = "1000" height = "500">

### Results

We have used kaggle GPU accelearator for model learning
and training . The IDE used are VS code,Jupyter and PyCharm
.The Packages used are tensorflow , pytorch, and OpenCV

Our modified model is successfully generating the multi segmented mask

<img src= images/Results.jpg  width = "800" height = "400">


Dataset: https://www.kaggle.com/datasets/shubhamsharma212001/testing2
