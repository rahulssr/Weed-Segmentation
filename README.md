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

<img src= images/dataset.jpg  width = "500" height = "500">

### Model Architecture


