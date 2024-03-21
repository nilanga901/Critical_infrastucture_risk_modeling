Critical_infrastucture_risk_modeling


Data Sources

1)Energy Infrastructure Data:
We use Energy Infrastructure data to collect the power plant locations in the north-east region. This is obtained from Homeland Infrastructure Foundation-Level Data. This consists of the Latitude and Longitutdes cooridantes of all power plants in north-east US including the generation type and more information.


2)Precipitation Data:
We use Precipitation data to collect the historical rain storm and snow storm data
in the north-east region. This is obtained from NASA POWER DATA [3]. This
consists of the historical precipitation information for each 0.25 Latitude and 0.25
Longitutdes coordinates of the northeast USA for the past 4 years.

3)Earthquake Data:
We use Earthquake Occurrence data to collect the historical intensity of earthquakes recorded in the north-east region.  This is obtained from U.S Geological Survey: Earthquake Hazard Program. This consists of the historical earthquake information for each occurred with relevant latitude and longitude coordinates of the northeast USA for the past 4 years.

4)Disaster Declarations Data:
We use Disaster Declarations data to collect the historical declarations of statewide disasters recorded in the north-east region. This is obtained from FEMA Web Disaster Declarations. This consists of the historical disaster declaration information for each that occurred with relevant diaster type at the northeast.

5)Power Disturbances:
We use power disturbances data to analyze how the historical disturbances have occurred in north-east region. This data is obtained from ISER Electric Disturbance Events. This consists of the historical Power Disturbances information for each event occurred.

6)Fire Occurrence
We use Fire Occurrence data to analyze how the historical wild and man-made
fires have occurred in the north-east region. This data is obtained from National
Interagency Fire Center. This consists of information on the location, fire type
and timeline for each event that occurred.


Constructed Datasets

Each of the following dataset is pre-processsed and constructed by calling precess_data.py. 

Dataset 1: 
We use Dataset 1 to encode geographical features to enable predictions of vulnerability conditioned on the geo-location. In this dataset we have the following
features.

i. Geographical Location -We consider a grid tiling of the north east region by 0.5 × 0.5 degrees of latitudes and longitudes. For each grid point, we evaluate the next features.

ii. State - We use the lat, long coordinate of each grid square to assign them to the corresponding state and we create a column with state information for each point.

iii. Generation Type - For each geo-location given by 0.5 × 0.5 square on the grid, we evaluate the power generation and assign the majority generation as the generation type of that block.

iv. Earthquake Risk -  For each geo-location given by 0.5 × 0.5 square on the grid, we evaluate whether an earthquake has happened at this location and assign 1 or 0 to create a binary
feature column.

v. Precipitation - For each geo-location given by 0.5 × 0.5 square on the grid, we evaluate the average daily precipitation to create a continuous feature vector. As the learning models we use are better suited for discrete variables, we quantize the continuous feature to bins created by the quartiles. 

vi. Fire Risk - For each geo-location given by 0.5 × 0.5 square on the grid, we evaluate the fire
risk by looking at the number of reported fires at each grid location historically. This creates a continuous feature vector. As the models we use are better suited for discrete variables, we quantize the continuous feature to bins created by the quartiles.


Dataset 2:

We use Dataset 2 to encode time-dependent seasonality into features to enable
predictions of hazards conditioned on the location given by state and the month of
the year. In this dataset, we have the following features.

i.Month - We consider the month of the year as the anchor point to evaluate the rest of the
features. This enables to look at the temporal and seasonal aspects of the data,
while Dataset 1 looks at the spatial distribution.

ii. State - For each of the following features, we use the corresponding state and create a
column with state information for each point as information such as disaster declarations are mostly state-wide.

iii. Disaster Types - From the disaster declrations from FEMA, we create the this feature. We also
add the corresponding month and the state to the dataset as well.

iv. Breakdown Rate - For each month and state combination, we evaluate the average breakdown rate
and add this as a feature. As the learning models we use are better suited for discrete variables, we quantize the continuous feature to bins created by the quartiles. 

v. Precipitation - For each month and state combination, we evaluate the average daily precipitation
to create a continuous feature vector. As the learning models we use are better suited for discrete variables, we quantize the continuous feature to bins created by the quartiles. 



Model Development

In this section we describe the model development for analysis. We use the following models for the two datasets.

For Dataset 1:
1. Learn conditional probability Distributions - Bayesian Network
2. Classify fire risk - Decision Tree
3. Classify fire risk - Naive Bayes Classifier
4. Classify fire risk - Deep Neural Network
5. Classify fire risk - Quantum Neural Network
6. Classify earthquake risk - Decision Tree
7. Classify earthquake risk - Naive Bayes Classifier
8. Classify earthquake risk - Deep Neural Network
9. Classify earthquake risk - Quantum Neural Network
For Dataset 2:
1. Learn conditional probability Distributions - Bayesian Network
2. Classify breakdown rate - Decision Tree
3. Classify breakdown rate - Naive Bayes Classifier
4. Classify breakdown rate - Deep Neural Network
5. Classify breakdown rate - Quantum Neural Network

Code for learning conditional probability Distributions with  Bayesian Network : bayesian_net.py
Code for all classifications : baselines_quantum_DT_NB_DNN.ipynb