**AML -
https://docs.google.com/document/d/126f6CAmwOU_xlaEpQA4YGY67LOH_rjBpD7RXxlz1EAs/edit?usp=drivesdk**

------------------------------------------------------------------------
[Open PDF Document (Raw Link)]([https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME/raw/main/1.pdf](https://github.com/codingmickeymouse/code/blob/main/docs/aml.pdf))

**1. Web Crawlers using BeautifulSoup, lxml, and Scrapy**

**Aim:**\
To extract structured data from websites using BeautifulSoup, lxml, and
Scrapy, and compare their approaches.

**Algorithm:**

- Send a request to the website.

- Parse the HTML content using a parser (lxml/BeautifulSoup).

- Locate desired tags/classes.

- Extract and store the data.

**Program:**

import requests

from bs4 import BeautifulSoup

url = \"https://example.com\"

response = requests.get(url)

soup = BeautifulSoup(response.content, \'lxml\')

data = \[\]

for article in soup.find_all(\"article\", class\_=\"news-item\"):

data.append({

\"title\": article.find(\"h2\").text,

\"summary\": article.find(\"p\").text

})

print(data)

**Output:**

\[

{\"title\": \"News 1\", \"summary\": \"Summary 1\"},

{\"title\": \"News 2\", \"summary\": \"Summary 2\"}

\]

------------------------------------------------------------------------

**2. Graph-based Covid-19 Data with NetworkX**

**Aim:**\
To visualize relationships in Covid-19 data using graphs.

**Algorithm:**

- Collect data: nodes = people/places, edges = interactions.

- Add nodes and edges to a graph.

- Analyze and visualize using NetworkX.

**Program:**

import networkx as nx

import matplotlib.pyplot as plt

G = nx.Graph()

G.add_edges_from(\[

(\"CityA\", \"Patient1\"),

(\"Patient1\", \"Patient2\"),

(\"CityB\", \"Patient3\"),

(\"Patient2\", \"Patient4\"),

(\"Patient3\", \"Patient1\")

\])

nx.draw(G, with_labels=True)

plt.show()

**Output:**\
A graph showing connections between cities and patients.

------------------------------------------------------------------------

**3. Wine Quality Prediction (ML Model)**

**Aim:**\
Predict wine quality based on features like acidity, sugar, alcohol,
etc.

**Algorithm:**

- Load data and split into train/test.

- Scale features.

- Train a classifier.

- Predict and evaluate accuracy.

**Program:**

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

\# Load dataset

X = df.drop(\"quality\", axis=1)

y = df\[\"quality\"\]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = SVC().fit(X_train, y_train)

predictions = model.predict(X_test)

**Output:**\
Wine quality predictions on test samples.

------------------------------------------------------------------------

**4. House Price Estimation (Regression Model)**

**Aim:**\
Estimate house prices using features like area, year, and size.

**Algorithm:**

- Load dataset.

- Preprocess and split data.

- Train regression model.

- Predict and evaluate.

**Program:**

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor().fit(X_train, y_train)

predictions = model.predict(X_test)

**Output:**\
Predicted house prices for the test set.

------------------------------------------------------------------------

**5. Movie Recommendation System**

**Aim:**\
Recommend movies based on user ratings using collaborative filtering.

**Algorithm:**

- Prepare user-movie rating matrix.

- Compute similarity between users/items.

- Recommend top movies using similarity scores.

**Program:**

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

\# Mock rating matrix

data = pd.DataFrame({

\"User1\": \[5, 3, 0, 0\],

\"User2\": \[4, 0, 0, 2\],

\"User3\": \[1, 1, 0, 5\]

})

sim = cosine_similarity(data.T)

**Output:**\
Similarity matrix, which helps in generating recommendations.

------------------------------------------------------------------------

**6. K-Means Clustering of Indian States**

**Aim:**\
Cluster Indian states based on socio-economic indicators.

**Algorithm:**

- Load state-wise feature data.

- Scale data.

- Apply KMeans.

- Assign and interpret clusters.

**Program:**

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)

kmeans.fit(X_scaled)

clusters = kmeans.labels\_

**Output:**\
Cluster labels assigned to each state.

------------------------------------------------------------------------
