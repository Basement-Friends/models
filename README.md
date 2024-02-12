Repository contains three machine learning models:
# age classificator # 
classifies age from the person's photo. It's deep learning model biult with TensorFlow. 

Model was trained on following datasets: 
- https://www.kaggle.com/datasets/frabbisw/facial-age
- https://www.kaggle.com/datasets/jangedoo/utkface-new
- https://www.kaggle.com/datasets/roshan81/ageutk

# face detector #
model detects human faces in the images. In the project YOLOv5s was used. 

Model was trained on following dataset: https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection/

# toxic messages detector #
model detects if message is toxic. This is voting classifier consisting of random forest, XGBoost and decision tree.

Model was trained on following datasets: 
- https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset?select=FinalBalancedDataset.csv
- https://www.kaggle.com/datasets/akashsuper2000/toxic-comment-classification?select=validation.csv

Model has about 76% accuracy on the test dataset.
