# Weed-Pro :  Precision Weed Detection and Segmentation 🌱🌱 

Weeds 🌿 are the unwanted guests in our fields. Not only do they hog resources, but they can also lead to a devastating scenario: nutrient deficiency in the early stages of a crop's life, eventually culminating in crop failure 🥀.

🎯 The Challenge: Weed management is tricky. It's not just about spotting a weed; it's about distinguishing between the main crop 🌾 and these unwelcome plants. Why? Because precise separation ensures effective management. But how do we tell them apart when, in nature, they look so similar?

Enter deep learning 🧠. With the power of advanced algorithms, we can now analyze images 📸, identify, and even segregate weeds from crops. This is revolutionary for precision agriculture, drastically reducing herbicide pollution and ensuring our crops get all the nourishment they need.

In this initiative, we shine the spotlight on onion-crop fields 🌰. The resemblance between crops and weeds in such fields is striking. So, our challenge is twofold:

Identify and separate weeds 🌿 from the main crop 🌾.
Distinguish weeds from the very soil they sprout from 🌍.
Our solution? A potent deep learning architecture. It doesn't just spot a weed; it segments it from an image, detects, and classifies it. At its core, this multiclass semantic segmentation fuses the strengths of both binary and multi-segmentation models.
<img src= images/Overview.png  width = "1000" height = "600">


## Dataset Collection 🔍🔍


Alright, let's give it another shot, incorporating emojis for visual emphasis:

🌞 Amidst the sun-kissed expanses of IGKV (Indira Gandhi Krishi Vishwavidyalaya), our cameras danced with the whispering winds. Each snap captured a unique tapestry; where the slender blades of monocot weeds swayed alongside the proud, earthy presence of the Onion (Allium cepa L.). 

<img src= images/dataset.jpg  width = "800" height = "400">

🌱 Venturing through these fields was more than just a collection process. It was a journey, a visual exploration of nature's intricate ballet, with each plant playing its part, waiting for its moment in the limelight. The challenge? To encapsulate this dance in a frame, preserving the authenticity of each performer.

🎨 Back in the dim ambiance of our workstations, the next phase commenced. With the 'LabelMe' tool as our digital brush, we painted annotations onto each image, distinguishing the nuanced tales of weeds from the unmistakable narrative of the onion. ![Space for the annotation process picture]

Collecting the dataset using python
```python
def draw_multi_masks(im, shape_dicts):
    
    blank = np.zeros(shape=im.shape, dtype=np.uint8)
    
    channels = []
    cls = [x['label'] for x in shape_dicts]
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
    label2poly = dict(zip(cls, poly))

    for i, label in enumerate(labels):
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], (hues[label], 255, 255))
            
    return cv2.cvtColor(blank, cv2.COLOR_HSV2RGB)
```

## 🎯 Binary Segmentation using UNET



In UNET's architecture, let's define:

E as the encoder layers
D as the decoder layers
S as the skip connections
The fundamental expression for the UNET architecture can be represented as:

$`U(E, D, S) = D(S(E) ⊕ E)`$





Where:

⊕ represents the concatenation operation of skip connections.
The skip connections, S, ensure that low-level features from the encoder are combined with the corresponding layers in the decoder. These connections are essential for capturing attributes like edges, color variations, and gradient orientations. They preserve details such as onion crop leaf textures, image characteristics, and nuances in shadows and contrast.

The chosen loss function for this model is the binary cross entropy (BCE), which can be expressed as:

$`L = −∑ 
i=1
n
​
 [y 
i
​
 ⋅log(p 
i
​
 )+(1−y 
i
​
 )⋅log(1−p 
i
​
 )]`$

<img src= images/unet.png  width = "1000" height = "500">

Where:

n is the total number of pixels in the image.
y_i is the true label for the i-th pixel (1 for foreground, 0 for background).
p_i is the predicted probability for the i-th pixel being foreground.
This representation ensures a clear understanding of how UNET leverages both the architecture and the loss function to produce precise binary segmentations.



```python
import tensorflow as tf

def unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # ...[more layers, omitted for brevity]...
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

# 📊 Segmentation Result:

# Binary Segmentation 🌓:
The binary segmentation model exhibited a robust ability to distinguish between the main crop and any other element, be it weeds or the soil itself. Our accuracy rates remained consistently high, demonstrating that this model can be trusted for tasks that require a simple foreground-background classification.



## 🎯 Multi-class Segmentation using UNET

Semantic image segmentation is pivotal in fields like precision agriculture, medical imaging, and intelligent transportation. Segmentation is generally split into binary and multi-class types. While binary segmentation classifies objects into a single group, multi-class segmentation discerns multiple classes from an image.

At the heart of the vanilla U-Net architecture are the skip connections. They link each encoder layer to its counterpart in the decoder section. These connections capture high-level attributes, such as edges, colors, and gradients. This ensures intricate details, like the unique texture of onion leaves or the nuances of shadows and contrasts, don't get lost during the segmentation process.

<img src= images/U2.png  width = "1000" height = "500">

Mathematically, the loss in the vanilla U-Net model is represented as:

$`J 
p
​
 =− 
N
1
​
 ∑ 
i=1
N
​
 ∑ 
j=1
M
​
 y 
ij
​
 log(p 
ij
​
 )`$

 In this equation:


J 
p
​
  denotes the overall loss.

K stands for the normalization factor (-1/N).

N is the number of observations.
U-Net's beauty lies in its ability to maintain spatial resolution through skip connections from multiple encoders. This boosts the quality of the resulting feature maps. After these connections, the architecture includes two 3x3 convolution operations, batch normalization, and ReLU activations. The architecture culminates in a convolution layer powered by a sigmoid activation function to produce the segmentation mask.

The loss for the Weed-Net can be given as:

$`J(y)=BCE 
B
​
 (y)+BCE 
M
​
 (y)`$

```python
import tensorflow as tf

def UNet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    # Decoder
    u2 = tf.keras.layers.UpSampling2D((2, 2))(p1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    
    # Multi-class output
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(c2)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = UNet((128, 128, 3), 3)  # Example for 128x128 RGB images and 3 classes.
```
# Multi-class Segmentation Result  🎨:
Diving deeper into the nuances of our fields, the multi-class segmentation model took on the challenge of distinguishing among several entities: the main crop, various types of weeds, and the soil. The results were promising, with precision and recall values indicating the model's prowess in differentiating even closely resembling plants. This level of granular segmentation ensures that weed management can be performed with unprecedented accuracy, leading to healthier crops and more sustainable farming practices.

In essence, while the binary model provided a reliable broad-stroke picture, the multi-class segmentation model catered to the intricacies, catering to the diverse needs of precision agriculture.


# Merging of ROI Extraction and Multi-class Predictions:

The integration of predictions from ROI extraction and multi-class segmentation is pivotal in refining the segmentation's accuracy and detail. We harness the distinct strengths of two models: the binary model 
�
�
�
�
b 
img
​
  adeptly identifies shapes, while the multi-class model 
�
�
�
�
m 
img
​
  excels at discerning color features. Both predictions have a size of 
256
×
256
256×256.

For each pixel location 
(
�
,
�
)
(i,j) across channels 
�
k (representing R, G, B), we blend the two images into a merged output 
�
�
�
�
o 
img
​
 . The merging logic is encapsulated in the following formula:

$`o 
img
​
 [i][j][k]={ 
m 
img
​
 [i][j][k]
0
​
  
if b 
img
​
 [i][j]

=0
otherwise`$
​

​
 
By this approach, only the regions detected by the binary model 
�
�
�
�
b 
img
​
  get the color attributes from 
�
�
�
�
m 
img
​
 . This effectively means "painting" the shape outlines from the binary model with the color information of the multi-class model. The synergy between the models provides a holistic representation, allowing for a rich and precise segmentation result.



### Overview of the Model



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






Our model is a combination of two UNET with certain skip connections illustrated in architecture The left part
of the model segments the image into binary segmentation
and acts as input for the right part of the model which then
multi segments the image and thus the combination of both is
formed
The loss function is BCE Loss of both the binary part and
multi part

J(y) = BCEB(y) + BCEM(y) 



### Results

We have used kaggle GPU accelearator for model learning
and training . The IDE used are VS code,Jupyter and PyCharm
.The Packages used are tensorflow , pytorch, and OpenCV

Our modified model is successfully generating the multi segmented mask

<img src= images/Results.jpg  width = "800" height = "400">


Dataset: https://www.kaggle.com/datasets/shubhamsharma212001/testing2
