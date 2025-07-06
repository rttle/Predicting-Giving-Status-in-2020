<img width="200" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/dbbeb760-7ac3-4d53-84ce-a08071725da1">

# Predicting Giving Status in 2020
This repository holds an attempt to apply machine learning on constituent data (biographical, giving history, engagement) to predict whether or not they will give in 2020, the data used is provided by APRA: https://github.com/majerus/apra_data_science_courses/tree/master?tab=readme-ov-file.



## Overview
This project takes constituent, donor and friends of an organization, information housed in 3 tables to create a single dataset used to train a model to predict whether a constituent gives in 2020 or not. The 3 tables of information are made up of a mixture of categorical and numerical features. This repository shows how this project was addressed as a binary classification problem, with data cleaning and feature engineering applied to the provided tables in preparation for training multiple models to predict Give in 2020 status. From the metrics calculated (accuracy, precision, recall, f1 score, and AUC-ROC), Decision Tree was the best performing across the board, besides for recall, with all metrics ranging from 0.75 to 0.78.



## Summary of Workdone
### Data
- Data: Type: Tabular
  - Input: 3 CSV file of features, output: Give in 2020 status created using giving_data_table
- Size: 
  - bio_data_table: 100,000 constituents, 14 features
  - giving_data_table: 378,001 gifts, 6 features
  - engagement_data_table: 100,000 constituents, 8 features
- Instances (Train, Test, Validation Split): 
  - Train: 70,000 constituents (70%)
  - Test: 30,000 constituents (30%)
  - Validation: Cross-Validation on Train set


 
### Preprocessing / Clean up
**Missing Values/Duplicates.** Missing values were dealt with using either logically understanding of the data (Example: constituent has no giving data, therefore does not give in 2020), filling with defaults used in the domain (Example: Deceased = No unless someone changes status in database), or using median/mean/mode to fill missing values with understanding that such use narrows the distribution of the feature.

**Dropped.** All ID columns were dropped from the dataset. Repetitive location columns were also dropped. There were also columns made purely for purposes of feature engineering other columns and were not included in the final dataset. All columns relating to engagement were also dropped due to time constraints preventing further investigation into how to make the engagement table usable despite significant missing values.

**Outliers.** Outliers were mainly addressed by categorizing numerical columns. 

**Feature Engineering.** Significant feature engineering was done to aggregate the giving data provided. Aggregation was grouped by an individual constituents, and new columns were made to summarize an individual constituent’s giving history (Number of Hard/Soft Credit gifts, Median/Mean Time between Gifts, Average Gift Amount). The target variable, Give in 2020, was also created through feature engineering based on Gift Date in the giving table. Constituent biographical information also had feature engineering, in particular age was calculated using the constituent’s birthday, along with pulling out the constituent’s birthday month.

**Encoding.** One Hot Encoding had to be done in preparation for training the machine learning algorithms.

**Normalization.** The scale of the numerical columns were widely ranged, so normalization was applied.



### Data Visualization
The figure below is a summary table of the dataset. Note that the categorical/numerical classification is initial determinations and changed as a better understanding of the dataset was reached. These initial determinations were made by setting definitions through use of functions. This table does include columns from the engagement table despite not being used in the final dataset. In particular, this table shows how much missing values there are in the engagmement table seen in the middle starting with 'last_contact' and ending with 'interests.'

<img width="600" alt="Screenshot 2025-07-06 at 11 46 04 AM" src="https://github.com/user-attachments/assets/2652de04-22e8-433d-afc5-f8842c927c2b" />
 
The figure below is a portion of a pairplot, which was meant to show relationships of the features through the scatterplots and the distribution of the target variable classes, Give in 2020. The pairplot also easily showed what should be categorical features when bars of data were shown in the scatterplots, like seen below for Number of Soft Credits (Number_SoftCr) and Number of Gifts. 

<img width="507" alt="Screenshot 2025-07-06 at 11 54 59 AM" src="https://github.com/user-attachments/assets/fb7fdc7e-5ae4-4220-af05-b8add4e65f43" />


The figure below is histograms of the numerical features. Of the columns, none show easily separatable distributions between status. Would like to see more distinct peaks around different centers based on status instead of both overlapping over similar centers. 

<img width="864" alt="Screenshot 2025-07-06 at 11 48 57 AM" src="https://github.com/user-attachments/assets/78ce30e8-5052-485a-a7de-ac888c8eb4b5" />

Below is a bar graph that shows the dataset is fairly balanced.

<img width="568" alt="Screenshot 2025-07-06 at 11 49 11 AM" src="https://github.com/user-attachments/assets/8837d176-6e9e-4012-a452-a42da27d6fee" />



### Problem Formulation
- Input / Output
  - Input: 4 numerical features, 10 categorical features
  - Output: Give in 2020 Status (1 = Gives in 2020, 0 = Does not Give in 2020)
- Models 
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
- Hyperparameters
  - The following were the hyperparameters used for K Nearest Neighbors, Decision Tree, and Random Forest, respectively. 

<img width="652" alt="Screenshot 2025-07-06 at 12 07 23 PM" src="https://github.com/user-attachments/assets/9e77358c-1ef6-4ed5-b799-a9b243e1c3b8" />



### Training
For initial models, the dataset was split 70% for training and 30% for testing. Multiple models were trained by using dictionaries of classifiers and parameter grids. Grid Search was used to perform hyperparameter tuning and perform cross validation. Calibrated Classifier Cross Validation was used to help smooth out ROC curves, an issue seen in the prototype stage. 
The main issue for model training was time constraints and the limitations of the computer used to train the models. Time constraints limited the amount of time dedicated to preprocessing and model fine tuning. The hardware limitation compounded the issue due to needing to limit the number of models trained to allow for troubleshooting without having to dedicate too much time each time a change was made to retrain the models.



### Performance Comparison
Multiple metrics were computed for the models, including: accuracy, precision, recall, F1, and AUC-ROC score. The score computed from the function within the classifier instance was also included. All were included in a comparison table; however, of importance is AUC-ROC score. Confusion matrix was also used to visualize model performance because it reflects the scores seen in the table.
Below is table of metrics for the models trained. From the table, Decision Tree performed the best across the board.

<img width="747" alt="Screenshot 2025-07-06 at 12 10 47 PM" src="https://github.com/user-attachments/assets/76e3b2bc-4777-4127-86b1-b91e2f446b0e" />

Below is the Confusion Matrix for the Decision Tree model. It shows how the model was able to predict both the negative and positive classification had a good rate, reflecting the numerical metrics seen in the table above.

<img width="522" alt="Screenshot 2025-07-06 at 12 10 57 PM" src="https://github.com/user-attachments/assets/cc506ff9-916a-4df7-970d-5400e07fad64" />


Below is the ROC curve for the Decision Tree model, showcasing its AUC-ROC score of 0.78.

<img width="561" alt="Screenshot 2025-07-06 at 12 11 12 PM" src="https://github.com/user-attachments/assets/0bbed883-27b0-46fb-8a35-ca39aac794d7" />



### Conclusions
Of the models trained, Decision Tree did the best at predicting whether a constituent will give in 2020.  



### Future Work
To achieve better results, more data preprocessing should be done, along with consideration of other machine learning algorithms. For data preprocessing, city was a column and categorizing that could provide useful data for modeling. Time constraints prevent a deeper look at the engagement_data_table, so deeper cleaning and preprocessing could also add to the model. Some of the numerical columns could have been categorized to better take care of outliers. Use of other models such as XGBoost and Neural Networks could see better results.



## How to reproduce results
To reproduce results, download the bio_data_table, giving_data_table, and engagement_data_table csv files from the linked Github repository. Then ensure that the Development_Preprocess.py file is downloaded from this repository and run the Development_Models.ipynb notebook also found in this repository.



## Overview of files in repository
- **Development_Preprocessing_Visualization.ipynb:** Notebook that takes the provided development tables and prepares it as a dataframe to be used to train models. Also creates tables and visualizations for data understanding.
- **Development_Prototype.ipynb:** Notebook that shows the initial attempt at modeling. Takes a dataset, trains multiple models, compares the models through a metrics table, and ROC Curves for the models.
- **Development_Models.ipynb:** Notebook that expands on the Prototype, resulting in stronger models. Takes a dataset, trains multiple models, compares the models through a metrics table, and ROC Curves for the models. These models underwent hyperparameter training.
- **Development_Preprocess.py:** Module created to wrap all preprocessing done to the dataset in the preprocessing notebook. 



## Data
Data is from the APRA Data Science Course. https://github.com/majerus/apra_data_science_courses/tree/master?tab=readme-ov-file



## Citations
-APRA, *APRA Data Science Courses *, WWW.GITHUB.COM, Jun. 10, 2020, https://github.com/majerus/apra_data_science_courses.
