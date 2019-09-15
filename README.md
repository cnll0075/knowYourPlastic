# Know Your Plastics
### An Image Recognition & Deep Learning application built on Plastic-Net Dataset

An environmental awareness mobile app that helps you to identify your plastic product and provide it's environmental impact along with a bunch of metrics that you would impact the choices you make in usage of plastics in daily life - in which you eat, drink, store, wear, and dispose. Using state-of-the-art Image Recognition and Deep Learning techniques, we have created an open source api that predicts the type of plastic product given it's image and provides various metadata related to the product in terms of environmental awareness. We use the same API to power this effective daily-usage mobile application. 


## Problem Statement 

Plastic makes up about 20% of landfill garbage, and less than 10% of them are recycled each year. Plastics are non-biodegradable; when degraded through chemical means or burning, they release toxic and carcinogenic substances, posing substantial threats to the environment. To understand and keep track of the plastic we use daily, we trained a deep learning model (transfer learning with ResNet 50) on over 10,000 images of 43 types of common plastic products. Through taking a photo or directly updating a photo from the album, the user will learn the type of plastic the product belong to, the properties and carbon footprint of the product and its potential damage to the environment.


## Tech and Design Stack 
<img src="./imgs/kyc_stack.png" width="900"/>

## PlasticNet - Image Dataset

We based our deep learning application on our self-collected Image Dataset - PlasticNet. We have open sourced this dataset and anyone could use it for Image Recognition Purposes. Please download the dataset from this link - https://drive.google.com/file/d/1E9R4fRIOrCZloTRwIqvLoJpi1sMNjALL/view?usp=sharing

Classes and Data Dictionary - https://drive.google.com/file/d/1mf61dPkVTYJFG6U1QkwjWJKwDo-FYK9U/view?usp=sharing
