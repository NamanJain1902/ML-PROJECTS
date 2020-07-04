import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## required functions ##
def get_title_from_index(index):
  return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
  return df[df.title == title]["index"].values[0]

def combine_features(row):
  try:
    return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
  except:
    print ("Error:", row)
#############################

## 1. Read C.S.V file
df = pd.read_csv("movie_dataset.csv")

## 2. Select features
features = ['keywords', 'cast', 'genres', 'director']

## 3. Create a column in df which combines all selected features

for feature in features:
  df[feature] = df[feature].fillna(' ')

df["combined_features"] = df.apply(combine_features, axis = 1)

# print(df["combined_features"].head())

## 4. Create count matrix from this new combined column

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

## 5. Compute the Cosine Similarity based on the count_matrix

cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = "Iron Man"

## 6. Get index of this movie from it's title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

## 7. Get a list of similar movies in descending order of similaruty score
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse  = True)

## 8. Print titles of first 50 movies
i = 1
for element in sorted_similar_movies:
  print(get_title_from_index(element[0]))
  i += 1
  if i > 50:
    break
