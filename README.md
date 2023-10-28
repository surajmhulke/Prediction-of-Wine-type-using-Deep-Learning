# Prediction-of-Wine-type-using-Deep-Learning
# Prediction-of-Wine-type-using-Deep-Learning

Prediction of Wine type using Deep Learning
 
We use deep learning for the large data sets but to understand the concept of deep learning, we use the small data set of wine quality. You can find the wine quality data set from the UCI Machine Learning Repository which is available for free. The aim of this article is to get started with the libraries of deep learning such as Keras, etc and to be familiar with the basis of neural network. 
About the Data Set : 
Before we start loading in the data, it is really important to know about your data. The data set consist of 12 variables that are included in the data. Few of them are as follows – 
 

Fixed acidity : The total acidity is divided into two groups: the volatile acids and the nonvolatile or fixed acids.The value of this variable is represented by in gm/dm3 in the data sets.
Volatile acidity: The volatile acidity is a process of wine turning into vinegar. In this data sets, the volatile acidity is expressed in gm/dm3.
Citric acid : Citric acid is one of the fixed acids in wines. It’s expressed in g/dm3 in the data sets.
Residual Sugar : Residual Sugar is the sugar remaining after fermentation stops, or is stopped. It’s expressed in g/dm3 in the data set.
Chlorides : It can be a important contributor to saltiness in wine. The value of this variable is represented by in gm/dm3 in the data sets.
Free sulfur dioxide : It is the part of the sulfur dioxide that is added to a wine. The value of this variable is represented by in gm/dm3 in the data sets.
Total Sulfur Dioxide : It is the sum of the bound and the free sulfur dioxide.The value of this variable is represented by in gm/dm3 in the data sets.
 

Step #1: Know your data.
Loading the data. 
 

# Import Required Libraries 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
 
# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')
  
First rows of `red`. 
 


# First rows of `red`
red.head()
Output: 
 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/c3cb7972-191a-4ffd-9926-59449da4d4f9)



  
Last rows of `white`. 
 

# Last rows of `white`
white.tail()
Output: 
 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/f3842c6a-4507-4038-ab4b-ad3458f38262)



  
Take a sample of five rows of `red`. 
 

# Take a sample of five rows of `red`
red.sample(5)
Output: 
 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/277eaca7-1b48-45e1-b90d-313d7b840cab)



Data description – 
 

# Describe `white`
white.describe()
Output: 
 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/5f806585-7973-4d7d-b282-968d173392ee)



Check for null values in `red`. 
 

# Double check for null values in `red`
pd.isnull(red)
Output: 
 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/6b6cb3f1-d296-469a-bbce-04e902bb41e6)



 

Step #2: Distribution of Alcohol.
Creating Histogram. 
 

![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/d1774a94-6fe2-4302-b8aa-d72ac00545bc)

# Create Histogram
fig, ax = plt.subplots(1, 2)
 
ax[0].hist(red.alcohol, 10, facecolor ='red',
              alpha = 0.5, label ="Red wine")
 
ax[1].hist(white.alcohol, 10, facecolor ='white',
           ec ="black", lw = 0.5, alpha = 0.5,
           label ="White wine")
 
fig.subplots_adjust(left = 0, right = 1, bottom = 0, 
               top = 0.5, hspace = 0.05, wspace = 1)
 
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_ylim([0, 1000])
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
 
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()
Output: 
![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/146c9098-2857-4b1a-b423-cedb2bb1e397)


  
Splitting the data set for training and validation. 
 

# Add `type` column to `red` with price one
red['type'] = 1
 
# Add `type` column to `white` with price zero
white['type'] = 0
 
# Append `white` to `red`
wines = red.append(white, ignore_index = True)
 
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
X = wines.ix[:, 0:11]
y = np.ravel(wines.type)
 
# Splitting the data set for training and validating 
X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size = 0.34, random_state = 45)
Step #3: Structure of Network
# Import `Sequential` from `keras.models`
from keras.models import Sequential
 
# Import `Dense` from `keras.layers`
from keras.layers import Dense
 
# Initialize the constructor
model = Sequential()
 
# Add an input layer
model.add(Dense(12, activation ='relu', input_shape =(11, )))
 
# Add one hidden layer
model.add(Dense(9, activation ='relu'))
 
# Add an output layer
model.add(Dense(1, activation ='sigmoid'))
 
# Model output shape
model.output_shape
 
# Model summary
model.summary()
 
# Model config
model.get_config()
 
# List all weight tensors
model.get_weights()
model.compile(loss ='binary_crossentropy', 
  optimizer ='adam', metrics =['accuracy'])
Output: 
 

![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/f288f46f-067a-478b-8208-d4c2b609ccda)

![image](https://github.com/surajmhulke/Prediction-of-Wine-type-using-Deep-Learning/assets/136318267/b53db888-eac4-4d18-8428-4fe84cee184a)

 

Step #4: Training and Prediction
 

# Training Model
model.fit(X_train, y_train, epochs = 3,
           batch_size = 1, verbose = 1)
  
# Predicting the Value
y_pred = model.predict(X_test)
print(y_pred)
Output: 
 
![Uploading image.png…]()

