# -*- coding: utf-8 -*-
"""streamlit_knn.py
A streamlit application of Concrete strength prediction
using k nearest neighbor algorithm
"""

import streamlit as st
import random
import matplotlib.pyplot as plt
# import pandas as pd
from typing import List, Dict, Tuple, Callable

@st.cache
def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

data = parse_data("concrete_compressive_strength.csv")

# Build the App here
st.title('Compressive Strength Prediction')
st.write('This app predicts the compressive strength of materials using the KNN algorithm')

st.sidebar.subheader('Model Parameters')
k = st.sidebar.slider('Number of neighbors (K)', 1, 10, 3)


st.sidebar.subheader('User Input Values')
cement = st.sidebar.slider('Cement', 100, 600, 280)
slag = st.sidebar.slider('Slag', 0, 400, 75)
ash = st.sidebar.slider('Ash', 0, 220, 54)
water = st.sidebar.slider('Water', 0, 250, 180)
plasticizer = st.sidebar.slider('Super Plasticizer', 0.0, 40.0, 6.2, step=0.2)
coarse = st.sidebar.slider('Coarse Aggregate', 750, 1200, 972)
fine = st.sidebar.slider('Fine Aggregate', 500, 1000, 775)
age = st.sidebar.slider('Age', 1, 365, 45)

# Create a data frame out of the user inputs
#df = pd.DataFrame(
df =             {'Cement': [cement], 
                  'Slag': [slag],
                  'Ash': [ash],
                  'Water': [water],
                  'SuperPlasticizer': [plasticizer],
                  'CoarseAgg': [coarse],
                  'FineAgg': [fine],
                  'Age': [age]
                  }
#                )

# show the dataframe
st.subheader('User Input Values')
# st.dataframe(df)  # Same as st.write(df)
st.table(df)

def euclidean_distance(xs1:List[float], xs2:List[float]) -> float:
    return sum(map(lambda val: (val[1] - val[0]) ** 2, zip(xs1, xs2))) ** 0.5

# kNN model
def knn(dataset: List[List[float]], query: List[float], k: int) -> float:
    distances = []
    for index, observation in enumerate(dataset):
        distance = euclidean_distance(xs1=query, xs2=observation)
        distances.append((distance, index))
        # print(distance, index, observation)
    distances.sort(key=lambda item: item[0])
    mean_y = sum(map(lambda val: dataset[val[1]][-1], distances[:k])) / k
    return mean_y


# make prediction
# pred = knn(data, df.iloc[0], k)
query = [cement, slag, ash, water, plasticizer, coarse, fine, age]
# make prediction using the query feature values
pred = knn(data, query, k)


st.subheader('kNN Prediction')
txt = "Predicted Material Strength = {prediction:.3f}"
st.write(txt.format(prediction = pred))


# plot histogram of euclidean distances
st.subheader('Distance Plots')
def get_all_distances(dataset: List[List[float]], query: List[float]) -> List[float]:
    distances = []
    for index, observation in enumerate(dataset):
        distance = euclidean_distance(xs1=query, xs2=observation)
        distances.append(distance)

    return distances

# compute distances to each datapoint in the training data from the query instance
query_distances = get_all_distances(data, query)

# mean values of each features
mean_query = [280, 75, 54, 180, 6.2, 972, 775, 45]
mean_query_distances = get_all_distances(data, mean_query)

fig = plt.figure(figsize = (8, 4))
plt.hist([query_distances, mean_query_distances], color=['r','b'], alpha=0.5)
plt.xlabel("Distance to training data points")
plt.ylabel("Number of datapoints")
plt.title("Histogram of distances to datapoints")
plt.legend(['User Input Values as Query', 'Mean Feature Values as Query'])
st.pyplot(fig)



# Add additional reference information
st.subheader('Reference Information')
"""There are 1,030 observations and each observation has 8 measurements. 
The data dictionary for this data set tells us the definitions of the individual variables (columns/indices):

| Index | Variable | Definition |
|-------|----------|------------|
| 0     | cement   | kg in a cubic meter mixture |
| 1     | slag     | kg in a cubic meter mixture |
| 2     | ash      | kg in a cubic meter mixture |
| 3     | water    | kg in a cubic meter mixture |
| 4     | superplasticizer | kg in a cubic meter mixture |
| 5     | coarse aggregate | kg in a cubic meter mixture |
| 6     | fine aggregate | kg in a cubic meter mixture |
| 7     | age | days |
| 8     | concrete compressive strength | MPa |

The target ("y") variable is a Index 8, concrete compressive strength in [Mega Pascals](https://en.wikipedia.org/wiki/Pascal_(unit)).
"""

