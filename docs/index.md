# **AML** 

 **Manual doc at the end** 

### **1\. Web Crawlers: BeautifulSoup, lxml, and Scrapy**

* **Aim:** To implement and compare Python-based web crawlers (BeautifulSoup, lxml, and Scrapy) for extracting structured data from online sources.  
* **Algorithm:**  
  * **BeautifulSoup/lxml:**  
    1. Send an HTTP GET request to the target URL to fetch the webpage content.  
    2. Parse the HTML/XML content using BeautifulSoup with an lxml parser.  
    3. Navigate the parsed tree structure to locate and extract the desired data elements (e.g., using tags, classes, IDs).  
    4. Store the extracted data in a structured format (e.g., list of dictionaries, CSV).  
  * **Scrapy:**  
    1. Define an "Item" to specify the structure of the data to be extracted.  
    2. Create a "Spider" that defines how to follow links (if necessary) and parse pages.  
    3. The Scrapy engine handles requests, sends them to the downloader, receives responses, and sends them to the spider for parsing.  
    4. The spider extracts data using selectors (CSS or XPath) and returns Items.  
    5. Items are processed through an Item Pipeline for cleaning, validation, and storage.  
* **Program (Conceptual):**  
  * **BeautifulSoup/lxml:**  
    \# import requests  
    \# from bs4 import BeautifulSoup  
    \#  
    \# url \= 'target\_website\_url' \# e.g., a mock news site  
    \# \# Mock HTML content for the example output:  
    \# \# mock\_html \= """  
    \# \# \<html\>\<body\>  
    \# \#   \<article class='news-item'\>\<h2\>Title 1\</h2\>\<p\>Summary 1\</p\>\</article\>  
    \# \#   \<article class='news-item'\>\<h2\>Title 2\</h2\>\<p\>Summary 2\</p\>\</article\>  
    \# \# \</body\>\</html\>  
    \# \# """  
    \# \# response \= requests.get(url) \# In a real scenario  
    \# \# soup \= BeautifulSoup(response.content, 'lxml') \# In a real scenario  
    \# \# For this example, let's assume soup is parsed from mock\_html  
    \# \# data\_elements \= soup.find\_all('article', class\_='news-item')  
    \# \# extracted\_data \= \[\]  
    \# \# for element in data\_elements:  
    \# \#     title \= element.find('h2').text  
    \# \#     summary \= element.find('p').text  
    \# \#     extracted\_data.append({'title': title, 'summary': summary})  
    \# \# print(extracted\_data)

  * **Scrapy:**  
    \# \# items.py  
    \# import scrapy  
    \# class ProductItem(scrapy.Item):  
    \#     name \= scrapy.Field()  
    \#     price \= scrapy.Field()  
    \#  
    \# \# spiders/my\_spider.py (Illustrative name)  
    \# import scrapy  
    \# \# from ..items import ProductItem \# Assuming items.py is in a parent directory or structured appropriately  
    \#  
    \# class MySpider(scrapy.Spider):  
    \#     name \= 'myspider'  
    \#     start\_urls \= \['target\_website\_url'\] \# e.g., a mock e-commerce site  
    \#  
    \#     def parse(self, response):  
    \#         \# \# Mock parsing logic for example output:  
    \#         \# \# products \= response.xpath("//div\[@class='product'\]")  
    \#         \# \# for product in products:  
    \#         \# \#     item \= ProductItem()  
    \#         \# \#     item\['name'\] \= product.xpath(".//h3/text()").get()  
    \#         \# \#     item\['price'\] \= product.xpath(".//span\[@class='price'\]/text()").get()  
    \#         \# \#     yield item  
    \#         pass \# Simplified for conceptual overview

* **Output:**  
  For **BeautifulSoup/lxml** (e.g., scraping news titles and summaries):  
  \[  
    {'title': 'Global Summit Addresses Climate Change', 'summary': 'Leaders from around the world gathered to discuss new initiatives.'},  
    {'title': 'Tech Advances in Renewable Energy', 'summary': 'Innovations in solar and wind power are paving the way for a greener future.'},  
    {'title': 'Local Library Announces Summer Reading Program', 'summary': 'The program aims to encourage reading among children and adults.'}  
  \]

  For **Scrapy** (e.g., scraping product names and prices from an e-commerce site, output typically goes to a file or database, here represented as console output for one item):  
  {'name': 'Wireless Headphones X200', 'price': '₹4999'}  
  {'name': 'Smartwatch Series 5', 'price': '₹12999'}

### **2\. Graph-based Representation of Covid-19 Data with NetworkX**

* **Aim:** To create a visual, graph-based representation of Covid-19 data using the NetworkX library in Python to analyze relationships and trends.  
* **Algorithm:**  
  1. **Data Acquisition:** Obtain Covid-19 data (e.g., cases, deaths, recoveries, patient connections, location data).  
  2. **Node Definition:** Identify entities to represent as nodes (e.g., individuals, locations, virus strains, articles).  
  3. **Edge Definition:** Determine relationships between nodes to represent as edges (e.g., contact between individuals, spread from one location to another, co-occurrence of terms in research papers).  
  4. **Graph Construction:** Use NetworkX to create a graph object and add the defined nodes and edges. Assign attributes to nodes/edges if necessary (e.g., number of cases for a location node, type of contact for an edge).  
  5. **Analysis & Visualization:** Use NetworkX functions to analyze graph properties (e.g., centrality, communities) and visualize the graph using libraries like Matplotlib to identify patterns and trends.  
* **Program (Conceptual):**  
  \# import networkx as nx  
  \# import matplotlib.pyplot as plt  
  \# import pandas as pd  
  \#  
  \# \# Mock data for conceptual program  
  \# \# data \= {'source': \['CityA', 'Patient1', 'CityB', 'Patient2', 'Patient3'\],  
  \# \#         'target': \['Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient1'\],  
  \# \#         'relationship\_type': \['transmission\_hub', 'contact', 'transmission\_hub', 'contact', 'contact'\],  
  \# \#         'weight': \[None, 1, None, 1, 1\]}  
  \# \# covid\_data\_df \= pd.DataFrame(data)  
  \#  
  \# \# G \= nx.Graph() \# Or DiGraph  
  \#  
  \# \# \# Add nodes with attributes  
  \# \# G.add\_node("CityA", type="location", cases=500)  
  \# \# G.add\_node("CityB", type="location", cases=300)  
  \# \# G.add\_node("Patient1", type="person", age=45, status="recovered")  
  \# \# G.add\_node("Patient2", type="person", age=30, status="active")  
  \# \# G.add\_node("Patient3", type="person", age=60, status="active")  
  \# \# G.add\_node("Patient4", type="person", age=22, status="recovered")  
  \#  
  \# \# \# Add edges from the DataFrame or manually  
  \# \# G.add\_edge("CityA", "Patient1", relation="first\_case\_source")  
  \# \# G.add\_edge("Patient1", "Patient2", relation="direct\_contact", date="2023-03-10")  
  \# \# G.add\_edge("CityB", "Patient3", relation="first\_case\_source")  
  \# \# G.add\_edge("Patient2", "Patient4", relation="household\_contact", date="2023-03-15")  
  \# \# G.add\_edge("Patient3", "Patient1", relation="travel\_contact", date="2023-03-05") \# Patient1 got re-exposed or different context  
  \#  
  \# \# \# Basic analysis  
  \# \# \# print(f"Number of nodes: {G.number\_of\_nodes()}")  
  \# \# \# print(f"Number of edges: {G.number\_of\_edges()}")  
  \# \# \# print(f"Neighbors of Patient1: {list(G.neighbors('Patient1'))}")  
  \#  
  \# \# \# Visualization (actual plot would be generated)  
  \# \# \# pos \= nx.spring\_layout(G, seed=42)  
  \# \# \# node\_colors \= \['blue' if G.nodes\[n\]\['type'\] \== 'location' else 'red' for n in G.nodes()\]  
  \# \# \# nx.draw(G, pos, with\_labels=True, node\_color=node\_colors, node\_size=2000, font\_size=10, font\_weight='bold')  
  \# \# \# edge\_labels \= nx.get\_edge\_attributes(G, 'relation')  
  \# \# \# nx.draw\_networkx\_edge\_labels(G, pos, edge\_labels=edge\_labels)  
  \# \# \# plt.title("COVID-19 Contact Tracing Network (Conceptual)")  
  \# \# \# plt.show()

* Output:  
  The program would typically generate a plot (image) of the network. A textual representation of some basic graph properties could be:  
  Number of nodes: 6  
  Number of edges: 5  
  Nodes: \['CityA', 'CityB', 'Patient1', 'Patient2', 'Patient3', 'Patient4'\]  
  Edges: \[('CityA', 'Patient1'), ('Patient1', 'Patient2'), ('Patient1', 'Patient3'), ('CityB', 'Patient3'), ('Patient2', 'Patient4')\]  
  Neighbors of Patient1: \['CityA', 'Patient2', 'Patient3'\]  
  Degree of Patient1: 3  
  (A visual graph plot would also be displayed, showing nodes and connections)

### **3\. Machine Learning Model for Wine Quality Prediction**

* **Aim:** To develop a machine learning model to predict the quality of wine based on its physicochemical properties (e.g., acidity, alcohol content, sugar levels).  
* **Algorithm:**  
  1. **Data Loading:** Load the wine dataset, which includes features like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, alcohol, etc., and a target variable 'quality' (typically on a scale, e.g., 0-10).  
  2. **Data Preprocessing:**  
     * Handle missing values (if any).  
     * Encode categorical features (if any).  
     * Scale numerical features (e.g., using StandardScaler) to bring them to a similar range, which can improve model performance.  
  3. **Train-Test Split:** Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.  
  4. **Model Selection & Training:** Choose a suitable classification algorithm (e.g., Support Vector Machines (SVM), Naive Bayes, Random Forest, Neural Network). Train the selected model using the training data.  
  5. **Prediction & Evaluation:** Use the trained model to predict wine quality on the test set. Evaluate the model's performance using metrics like accuracy, precision, recall, F1-score, or confusion matrix.  
* **Program (Conceptual):**  
  \# import pandas as pd  
  \# from sklearn.model\_selection import train\_test\_split  
  \# from sklearn.preprocessing import StandardScaler  
  \# from sklearn.svm import SVC \# Example classifier  
  \# from sklearn.metrics import accuracy\_score, classification\_report  
  \#  
  \# \# Assume 'wine\_quality\_data.csv' exists with columns like:  
  \# \# 'fixed\_acidity', 'volatile\_acidity', ..., 'alcohol', 'quality'  
  \# \# For example:  
  \# \# wine\_data \= {  
  \# \#     'fixed\_acidity': \[7.4, 7.8, 7.8, 11.2\], 'volatile\_acidity': \[0.70, 0.88, 0.76, 0.28\],  
  \# \#     'citric\_acid': \[0.00, 0.00, 0.04, 0.56\], 'residual\_sugar': \[1.9, 2.6, 2.3, 1.9\],  
  \# \#     'chlorides': \[0.076, 0.098, 0.092, 0.075\], 'alcohol': \[9.4, 9.8, 9.8, 9.8\],  
  \# \#     'quality': \[5, 5, 5, 6\]  
  \# \# }  
  \# \# data \= pd.DataFrame(wine\_data)  
  \# \# X \= data.drop('quality', axis=1)  
  \# \# y \= data\['quality'\]  
  \#  
  \# \# X\_train, X\_test, y\_train, y\_test \= train\_test\_split(X, y, test\_size=0.25, random\_state=42) \# Assuming 4 samples, 1 for test  
  \#  
  \# \# scaler \= StandardScaler()  
  \# \# X\_train\_scaled \= scaler.fit\_transform(X\_train)  
  \# \# X\_test\_scaled \= scaler.transform(X\_test)  
  \#  
  \# \# model \= SVC(kernel='linear', random\_state=42) \# Using linear for simplicity with few samples  
  \# \# model.fit(X\_train\_scaled, y\_train)  
  \#  
  \# \# predictions \= model.predict(X\_test\_scaled)  
  \# \# accuracy \= accuracy\_score(y\_test, predictions)  
  \# \# report \= classification\_report(y\_test, predictions, zero\_division=0) \# zero\_division for small sample case  
  \#  
  \# \# print(f"Accuracy: {accuracy:.4f}")  
  \# \# print("Classification Report:\\n", report)  
  \#  
  \# \# \# Example prediction for a new wine sample:  
  \# \# \# new\_wine\_sample \= pd.DataFrame({  
  \# \# \#    'fixed\_acidity': \[7.0\], 'volatile\_acidity': \[0.60\], 'citric\_acid': \[0.10\],  
  \# \# \#    'residual\_sugar': \[2.0\], 'chlorides': \[0.080\], 'alcohol': \[10.0\]  
  \# \# \# })  
  \# \# \# new\_wine\_sample\_scaled \= scaler.transform(new\_wine\_sample)  
  \# \# \# predicted\_quality \= model.predict(new\_wine\_sample\_scaled)  
  \# \# \# print(f"Predicted quality for new sample: {predicted\_quality\[0\]}")

* Output:  
  Based on a hypothetical run with a small dataset and a split:  
  Accuracy: 0.8571  
  Classification Report:  
                 precision    recall  f1-score   support

             5       0.80      1.00      0.89         4  
             6       1.00      0.67      0.80         3  
             7       0.00      0.00      0.00         0

      accuracy                           0.86         7  
     macro avg       0.60      0.56      0.56         7  
  weighted avg       0.89      0.86      0.85         7

  Predicted quality for a new sample: 6

  *(Note: The classification report quality depends heavily on the dataset size and diversity. The example above is illustrative for a small, possibly imbalanced test set where some classes might not be present or predicted correctly, hence the zero\_division=0 in the conceptual code).*

### **4\. Regression Model for House Price Estimation**

* **Aim:** To build a regression model to estimate house prices using features like lot area, year built, basement size, and above-ground living area.  
* **Algorithm:**  
  1. **Data Loading:** Load the housing dataset, which includes features like LotArea, YearBuilt, TotalBsmtSF, GrLivArea, and the target variable SalePrice.  
  2. **Data Preprocessing:**  
     * Handle missing values (e.g., imputation with mean, median, or a constant).  
     * Encode categorical features (e.g., one-hot encoding for Neighborhood, HouseStyle).  
     * Feature scaling (e.g., StandardScaler or MinMaxScaler) for numerical features.  
     * Feature engineering (e.g., creating new features like AgeOfHouse \= CurrentYear \- YearBuilt, TotalSF \= TotalBsmtSF \+ GrLivArea).  
  3. **Train-Test Split:** Split the dataset into training and testing sets.  
  4. **Model Selection & Training:** Choose a suitable regression algorithm (e.g., Linear Regression, Ridge Regression, Lasso Regression, Random Forest Regressor, Gradient Boosting Regressor). Train the model on the training data.  
  5. **Prediction & Evaluation:** Use the trained model to predict house prices on the test set. Evaluate the model using regression metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).  
* **Program (Conceptual):**  
  \# import pandas as pd  
  \# from sklearn.preprocessing import StandardScaler  
  \# from sklearn.cluster import KMeans  
  \# import matplotlib.pyplot as plt  
  \#  
  \# \# Load dataset (assuming 'indian\_states\_socioeconomic.csv' with 'State' and other features)  
  \# \# data \= pd.read\_csv('indian\_states\_socioeconomic.csv')  
  \# \# \# Assume 'State' column exists, and others are numerical features  
  \# \# features\_to\_cluster \= \['Population', 'LiteracyRate', 'UnemploymentRate', 'UrbanRatio'\]  
  \# \# X \= data\[features\_to\_cluster\].copy() \# Create a copy to avoid SettingWithCopyWarning  
  \#  
  \# \# \# Handle missing values (example: mean imputation)  
  \# \# \# for col in features\_to\_cluster:  
  \# \# \#     X\[col\].fillna(X\[col\].mean(), inplace=True)  
  \#  
  \# \# scaler \= StandardScaler()  
  \# \# X\_scaled \= scaler.fit\_transform(X)  
  \#  
  \# \# \# Determine optimal K (e.g., using Elbow method)  
  \# \# \# wcss \= \[\]  
  \# \# \# for i in range(1, 11):  
  \# \# \#     kmeans\_elbow \= KMeans(n\_clusters=i, init='k-means++', max\_iter=300, n\_init=10, random\_state=0)  
  \# \# \#     kmeans\_elbow.fit(X\_scaled)  
  \# \# \#     wcss.append(kmeans\_elbow.inertia\_)  
  \# \# \# plt.plot(range(1, 11), wcss)  
  \# \# \# plt.title('Elbow Method')  
  \# \# \# plt.xlabel('Number of clusters')  
  \# \# \# plt.ylabel('WCSS')  
  \# \# \# plt.show()  
  \#  
  \# \# \# Based on Elbow method, let's assume optimal K is 3  
  \# \# optimal\_k \= 3  
  \# \# kmeans \= KMeans(n\_clusters=optimal\_k, init='k-means++', max\_iter=300, n\_init=10, random\_state=0)  
  \# \# data\['Cluster'\] \= kmeans.fit\_predict(X\_scaled)  
  \#  
  \# \# \# Analyze cluster characteristics  
  \# \# cluster\_summary \= data.groupby('Cluster')\[features\_to\_cluster\].mean()  
  \# \# print("Cluster Summary (Mean Values):")  
  \# \# print(cluster\_summary)  
  \#  
  \# \# \# To see which states fall into which cluster:  
  \# \# \# for i in range(optimal\_k):  
  \# \# \#     print(f"\\nStates in Cluster {i}:")  
  \# \# \#     print(data\[data\['Cluster'\] \== i\]\['State'\].values)  
  \# \# Simplified for conceptual overview

* **Output:**  
  * Assignment of each Indian state to one of K clusters.  
  * The characteristics (mean values of socio-economic features) of each cluster, allowing for interpretation of what defines each segment (e.g., "High Population, Low Literacy Cluster", "Low Unemployment, High Urbanization Cluster"). This helps in identifying patterns and disparities among states. A visualization (e.g., a scatter plot if using 2-3 key dimensions, or a table) of states and their assigned clusters.

