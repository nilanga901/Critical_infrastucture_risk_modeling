# Critical Infrastructure Risk Modeling

This repository contains code and data for modeling and analyzing risks to critical infrastructure in the northeastern United States. The project utilizes various data sources, including energy infrastructure, precipitation, earthquake, disaster declarations, power disturbances, and fire occurrence data.

## Data Sources

1. **Energy Infrastructure Data**: Obtained from the Homeland Infrastructure Foundation-Level Data, this dataset includes the locations (latitude and longitude coordinates) of power plants in the northeastern United States, along with information on generation type.

2. **Precipitation Data**: Historical precipitation data (rain and snow storms) for the northeastern United States, obtained from NASA POWER DATA. This dataset provides precipitation information for each 0.25° latitude and 0.25° longitude coordinate in the region for the past four years.

3. **Earthquake Data**: Historical earthquake intensity data for the northeastern United States, obtained from the U.S. Geological Survey's Earthquake Hazard Program. This dataset includes earthquake information with relevant latitude and longitude coordinates for the past four years.

4. **Disaster Declarations Data**: Historical statewide disaster declarations for the northeastern United States, obtained from the FEMA Web Disaster Declarations. This dataset includes information on disaster types and locations.

5. **Power Disturbances**: Historical power disturbance events in the northeastern United States, obtained from the ISER Electric Disturbance Events dataset.

6. **Fire Occurrence**: Historical data on wild and man-made fires in the northeastern United States, obtained from the National Interagency Fire Center. This dataset includes information on fire locations, types, and timelines.

## Constructed Datasets

The following datasets are constructed by running the `process_data.py` script:

### Dataset 1

This dataset encodes geographical features to enable predictions of vulnerability conditioned on geo-location. It includes the following features:

- Geographical Location: A grid tiling of the northeastern region by 0.5° × 0.5° latitude and longitude.
- State: The state corresponding to each grid square's location.
- Generation Type: The majority power generation type for each grid square.
- Earthquake Risk: A binary feature indicating whether an earthquake has occurred at each grid location.
- Precipitation: Average daily precipitation for each grid square, quantized into bins created by quartiles.
- Fire Risk: The number of reported fires at each grid location, quantized into bins created by quartiles.

### Dataset 2

This dataset encodes time-dependent seasonality into features to enable predictions of hazards conditioned on the location (state) and month of the year. It includes the following features:

- Month: The month of the year used as an anchor point for evaluating other features.
- State: The state corresponding to each data point.
- Disaster Types: The types of disasters declared in each state and month, obtained from FEMA disaster declarations.
- Breakdown Rate: The average breakdown rate for each state and month combination, quantized into bins created by quartiles.
- Precipitation: The average daily precipitation for each state and month combination, quantized into bins created by quartiles.

## Model Development

The following models are developed and used for analysis:

**For Dataset 1:**

1. Bayesian Network: Learn conditional probability distributions.
2. Decision Tree Classifier: Classify fire risk.
3. Naive Bayes Classifier: Classify fire risk.
4. Deep Neural Network: Classify fire risk.
5. Quantum Neural Network: Classify fire risk.
6. Decision Tree Classifier: Classify earthquake risk.
7. Naive Bayes Classifier: Classify earthquake risk.
8. Deep Neural Network: Classify earthquake risk.
9. Quantum Neural Network: Classify earthquake risk.

**For Dataset 2:**

1. Bayesian Network: Learn conditional probability distributions.
2. Decision Tree Classifier: Classify breakdown rate.
3. Naive Bayes Classifier: Classify breakdown rate.
4. Deep Neural Network: Classify breakdown rate.
5. Quantum Neural Network: Classify breakdown rate.

The code for learning conditional probability distributions with the Bayesian Network is available in `bayesian_net.py`. The code for all classification tasks is available in `baselines_quantum_DT_NB_DNN.ipynb`.