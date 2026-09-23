# Diabetes Progression Prediction Using Artificial Neural Network

## Project Overview

This project develops an **Artificial Neural Network (ANN)** regression
model to predict diabetes disease progression using the **Diabetes
dataset available in Scikit-learn**.

The notebook follows a complete deep learning workflow:

1.  Dataset loading
2.  Data inspection and preprocessing
3.  Exploratory Data Analysis (EDA)
4.  Train-test splitting
5.  Feature standardization
6.  Basic ANN development
7.  ANN training and evaluation
8.  Improved ANN development
9.  Model comparison
10. Final performance analysis

The project demonstrates how neural-network architecture and
hyperparameter changes can affect regression performance.

------------------------------------------------------------------------

## Objective

The objective is to model diabetes disease progression using the
available independent variables and evaluate whether an Artificial
Neural Network can learn useful relationships between the input features
and the target variable.

The project specifically focuses on:

-   Understanding the dataset
-   Preparing the data for neural-network training
-   Exploring relationships between variables
-   Building a basic ANN regression model
-   Evaluating the model using regression metrics
-   Improving the ANN architecture
-   Comparing the initial and improved models

> **Note:** The model is developed as an educational machine-learning
> project. Its results should not be interpreted as a clinically
> validated system or as a basis for medical diagnosis or treatment
> decisions.

------------------------------------------------------------------------

## Dataset

The project uses the `Diabetes` dataset from Scikit-learn:

``` python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
```

The dataset contains:

-   **442 observations**
-   **10 independent variables**
-   **1 continuous target variable**

### Features

The ten input features are:

-   `age`
-   `sex`
-   `bmi`
-   `bp`
-   `s1`
-   `s2`
-   `s3`
-   `s4`
-   `s5`
-   `s6`

The target represents a quantitative measure of diabetes disease
progression.

------------------------------------------------------------------------

## Technologies and Libraries

The notebook uses the following Python libraries:

  -----------------------------------------------------------------------
  Library                             Purpose
  ----------------------------------- -----------------------------------
  NumPy                               Numerical operations

  Pandas                              Data manipulation and analysis

  Matplotlib                          Data visualization

  Seaborn                             Statistical visualization

  Scikit-learn                        Dataset, preprocessing, train-test
                                      split and evaluation

  TensorFlow / Keras                  ANN model development and training
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Project Workflow

``` text
Scikit-learn Diabetes Dataset
            ↓
      Data Inspection
            ↓
   Missing Value / Duplicate Check
            ↓
     Exploratory Data Analysis
       ├── Target Distribution
       ├── Feature Distributions
       ├── Correlation Heatmap
       └── Feature vs Target Plots
            ↓
      Train-Test Split
          80 / 20
            ↓
     StandardScaler
            ↓
        Basic ANN
            ↓
       Model Training
            ↓
         Evaluation
     MSE / RMSE / MAE / R²
            ↓
       Improved ANN
            ↓
       Model Training
            ↓
         Evaluation
            ↓
      Model Comparison
```

------------------------------------------------------------------------

# 1. Data Loading and Preprocessing

The Diabetes dataset is loaded using `load_diabetes()`.

The notebook performs basic data inspection using:

-   `info()`
-   `describe()`
-   Missing-value checking
-   Duplicate-row checking

The input features and target variable are separated before model
training.

### Train-Test Split

The dataset is divided using:

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates:

-   **80% training data**
-   **20% testing data**

The test set is kept separate for evaluating model performance on unseen
data.

### Feature Scaling

`StandardScaler` is used to standardize the input features.

``` python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the
test data.

------------------------------------------------------------------------

# 2. Exploratory Data Analysis

The notebook performs EDA to understand the characteristics of the
dataset.

### Target Distribution

A histogram with KDE is used to visualize the distribution of the
diabetes progression target.

### Feature Distributions

Histograms are generated for the numerical variables to understand their
distributions.

### Correlation Analysis

A correlation heatmap is used to examine relationships between the
independent variables and the target.

### Feature vs Target Relationships

Scatter plots are generated for each input feature against the diabetes
progression target.

These visualizations provide an understanding of the relationships
present in the dataset before model development.

------------------------------------------------------------------------

# 3. Basic ANN Model

The initial ANN uses the following architecture:

``` text
Input Layer
    ↓
Dense Layer: 32 neurons, ReLU
    ↓
Output Layer: 1 neuron, Linear
```

### Model Configuration

  Parameter           Value
  ------------------- --------
  Hidden layers       1
  Hidden neurons      32
  Hidden activation   ReLU
  Output neurons      1
  Output activation   Linear
  Optimizer           Adam
  Learning rate       0.001
  Loss function       MSE
  Metric              MAE
  Epochs              100
  Batch size          16
  Validation split    20%

Because the target is continuous, this is treated as a **regression
problem**.

The output layer uses a linear activation function so that the network
can produce continuous numerical predictions.

------------------------------------------------------------------------

# 4. Basic ANN Results

The basic ANN produced the following results on the test dataset:

  Metric       Basic ANN
  ---------- -----------
  MSE          5266.8966
  RMSE           72.5734
  MAE            56.5354
  R² Score        0.0059

The very low R² score indicates that the initial architecture was not
able to explain much of the variation in the unseen test data.

This provided a baseline for further model improvement.

------------------------------------------------------------------------

# 5. Improved ANN Model

A deeper architecture was developed to improve the baseline model.

### Architecture

``` text
Input Layer
    ↓
Dense: 64 neurons, ReLU
    ↓
Dense: 32 neurons, ReLU
    ↓
Dropout: 20%
    ↓
Dense: 16 neurons, ReLU
    ↓
Output: 1 neuron, Linear
```

### Changes Introduced

Compared with the basic ANN, the improved model:

-   Increased the network depth
-   Increased the first hidden layer to 64 neurons
-   Added additional hidden layers with 32 and 16 neurons
-   Added a 20% dropout layer
-   Reduced the learning rate
-   Increased the maximum number of epochs

### Improved Model Configuration

  Parameter               Basic ANN   Improved ANN
  --------------------- ----------- --------------
  Hidden architecture            32   64 → 32 → 16
  Hidden activation            ReLU           ReLU
  Dropout                      None           0.20
  Learning rate               0.001         0.0005
  Epochs                        100            150
  Batch size                     16             16
  Loss                          MSE            MSE
  Optimizer                    Adam           Adam

------------------------------------------------------------------------

# 6. Improved ANN Results

The improved ANN achieved:

  Metric        Improved ANN
  ---------- ---------------
  MSE          **2842.6412**
  RMSE           **53.3164**
  MAE            **42.2657**
  R² Score        **0.4635**

------------------------------------------------------------------------

# 7. Model Comparison

The two models were compared using the same test dataset.

  Metric       Basic ANN    Improved ANN
  ---------- ----------- ---------------
  MSE          5266.8966   **2842.6412**
  RMSE           72.5734     **53.3164**
  MAE            56.5354     **42.2657**
  R² Score        0.0059      **0.4635**

### Observed Improvement

The improved ANN reduced:

-   MSE by approximately **46.0%**
-   RMSE by approximately **26.5%**
-   MAE by approximately **25.2%**

The R² score increased from approximately **0.006 to 0.4635**.

This indicates that the improved architecture learned substantially more
useful patterns from the available input variables than the initial
model.

------------------------------------------------------------------------

# 8. Interpretation of the Final Model

The improved ANN achieved an R² score of **0.4635** on the test dataset.

This means that the model explains approximately **46.3% of the
variation in the test-set diabetes progression values**.

The RMSE of approximately **53.32** and MAE of approximately **42.27**
indicate that prediction errors remain significant.

Therefore, the model demonstrates useful predictive capability for this
educational experiment, but the results also show that the model has
considerable room for further improvement.

The improvement from the basic model demonstrates that changing the
architecture, regularization and learning rate can substantially affect
ANN performance.

------------------------------------------------------------------------

# 9. Conclusion

The project successfully implemented an end-to-end ANN regression
workflow for predicting diabetes disease progression.

The initial ANN with a single hidden layer achieved an R² score of only
**0.0059**. An improved multi-layer architecture with **64, 32 and 16
neurons**, ReLU activation, **20% dropout**, and a lower learning rate
of **0.0005** produced substantially better results.

The improved model reduced MSE from **5266.90 to 2842.64**, RMSE from
**72.57 to 53.32**, and MAE from **56.54 to 42.27**. The R² score
increased from **0.0059 to 0.4635**.

These results demonstrate that the improved ANN was able to learn more
meaningful relationships between the available features and diabetes
progression than the baseline architecture.

However, the R² score of 0.4635 also indicates that more than half of
the variation in the test-set target remains unexplained by this model.
Further experimentation with architecture, hyperparameters,
regularization, feature engineering, and alternative regression
techniques could therefore be explored to improve performance.

Overall, the project demonstrates the complete deep-learning workflow
from preprocessing and EDA through ANN development, training,
evaluation, and iterative model improvement.

------------------------------------------------------------------------

## Future Improvements

Potential next steps include:

1.  Hyperparameter tuning for the number of neurons and hidden layers.
2.  Testing different learning rates and batch sizes.
3.  Applying early stopping.
4.  Experimenting with different regularization strategies.
5.  Comparing ANN performance with traditional regression models.
6.  Performing cross-validation where appropriate.
7.  Investigating feature engineering and feature selection.
8.  Testing alternative neural-network architectures.

------------------------------------------------------------------------

## Project Structure

A recommended repository structure is:

``` text
Diabetes-Progression-Prediction/
│
├── Diabetes Progression Prediction.ipynb
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

## Requirements

A typical environment for running the notebook includes:

``` text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

Install the required packages with:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

------------------------------------------------------------------------

## How to Run

1.  Clone or download the project repository.
2.  Install the required Python libraries.
3.  Open Jupyter Notebook or JupyterLab.
4.  Open:

``` text
Diabetes Progression Prediction.ipynb
```

5.  Run the notebook cells sequentially.
6.  Review the EDA visualizations, training curves, evaluation metrics
    and model comparison.

------------------------------------------------------------------------

## Disclaimer

This project is intended for **educational and machine-learning
demonstration purposes**. The model has not been clinically validated
and should not be used to diagnose diabetes, predict an individual
patient's medical outcome, or make treatment decisions.

------------------------------------------------------------------------

## Author

**Athul C Ravindran**

Data Science / AI & Machine Learning Project
