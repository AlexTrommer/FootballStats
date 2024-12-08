# Introduction
I am Alex Trommer, and I am a Junior studying Data Science at the University Of Michigan. I am a huge fan of football (soccer), so I chose to investigate upon a data set of football players and their various statistics, such as goals, age, 90s played (90 minutes played), shots on target, nation, position, as well as their teams, the league they play in, and for what season this is representing. Below is a sample of the dataset. This dataset has 88,310 rows and 10 columns, however only a fraction of these rows will actually be used.


| Season    | League   | Team           | Player         | Nation   | Position   |   Age |   90s |   Goals |   Shots On Target |
|:----------|:---------|:---------------|:---------------|:---------|:-----------|------:|------:|--------:|------------------:|
| 2000-2001 | EPL      | Manchester Utd | Gary Neville   | eng ENG  | DF         |    25 |  31.7 |       1 |                 5 |
| 2000-2001 | EPL      | Manchester Utd | Fabien Barthez | fr FRA   | GK         |    29 |  29.7 |       0 |                 0 |
| 2000-2001 | EPL      | Manchester Utd | David Beckham  | eng ENG  | MF         |    25 |  29.4 |       9 |                34 |
| 2000-2001 | EPL      | Manchester Utd | Paul Scholes   | eng ENG  | MF         |    25 |  27.2 |       6 |                18 |
| 2000-2001 | EPL      | Manchester Utd | Roy Keane      | ie IRL   | MF         |    28 |  26.4 |       2 |                15 |

The question I have that I will be looking to answer is "Can I build a model to accurately predict whether or not a player won the league in a given year?

# Data Cleaning and Exploratory Data Analysis

To kick things off (haha), we have to clean the data:
1:   I filled NA goals with 0
2:   I renamed some columns that were mispelled
3:   I dropped a couple of rows that had data we will not be using (such as squad total)

Out of curiosity, let's see who has scored the most amount of goals in a single season! I will find the highest goal tally and get the index!

| Player       | Team      | Season    | Position   |   Age |   Goals |   Shots On Target |
|:-------------|:----------|:----------|:-----------|------:|--------:|------------------:|
| Lionel Messi | Barcelona | 2011-2012 | FW,MF      |    24 |      50 |               114 |

Unsurprising to me, given it is Messi, but interesting nonetheless. 


What about finding the top 50 scorers per year? Well this is pretty simple to accomplish. By filtering the data and grouping the season, this is essentially what we get! (first 10 rows)
|                     | Player                  | League     | Team          |   Age | Position   |   Goals |   Shots On Target |
|:--------------------|:------------------------|:-----------|:--------------|------:|:-----------|--------:|------------------:|
| ('2000-2001', 2218) | Hernán Crespo           | SerieA     | Lazio         |    25 | FW         |      26 |                56 |
| ('2000-2001', 2691) | Mateja Kežman           | Eredivisie | PSV Eindhoven |    21 | FW         |      24 |                65 |
| ('2000-2001', 1100) | Raúl                    | LaLiga     | Real Madrid   |    23 | FW         |      24 |                58 |
| ('2000-2001', 2302) | Andriy Shevchenko       | SerieA     | Milan         |    23 | FW         |      24 |                55 |
| ('2000-2001', 1178) | Rivaldo                 | LaLiga     | Barcelona     |    28 | FW,MF      |      23 |                62 |
| ('2000-2001', 136)  | Jimmy Floyd Hasselbaink | EPL        | Chelsea       |    28 | FW         |      23 |                53 |
| ('2000-2001', 1346) | Javi Moreno             | LaLiga     | Alavés        |    25 | FW         |      22 |                53 |
| ('2000-2001', 614)  | Ebbe Sand               | Bundesliga | Schalke 04    |    28 | FW         |      22 |                61 |
| ('2000-2001', 2396) | Enrico Chiesa           | SerieA     | Fiorentina    |    29 | FW         |      22 |                74 |
| ('2000-2001', 916)  | Sergej Barbarez         | Bundesliga | Hamburger SV  |    28 | FW,MF      |      22 |                47 |

And here as a scatterplot:

<iframe
  src="Data/PlayerPlot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


However, as you can see along the y-axis, there is an issue with some of the data.
Let's take a step back. In soccer, a player has to get a shot on target in order to score. Realistically, it makes no sense. This must have gotten replaced during my data cleaning -- likely an NA replaced as a 0.

| League       |   0 |
|:-------------|----:|
| Bundesliga   |  18 |
| EPL          |  26 |
| Eredivisie   |  32 |
| LaLiga       |  11 |
| Ligue1       |  61 |
| PrimeiraLiga | 452 |
| SerieA       |  36 |

That is the distribution of cases in which the goals > 1 but shots on target are 0. It is a widespread issue!

We do imputation based off of the league -- a linear regression model!

After imputation, we filter the data as such:
1:   Players who contain "FW" in their position
2:   Played at least 5 90s (Ensures substitute players don't influence the data too much with low goals shots on target if they win the league)
3:   Data from only the last five years

Also, in the 2019-2020 season, the eredivisie was suspended because of the covid pandemic, so to be safe let's drop the entire year.

Finally it's time to add the new column, which will be used to train the data for a K-Nearest-Neighbor classifier prediction model. Let's make a dictionary for the winners of the leagues for the last five years.

|              | 2022-2023        | 2021-2022        | 2020-2021        | 2019-2020       | 2018-2019       |
|--------------|------------------|------------------|------------------|-----------------|-----------------|
| EPL          | Manchester City  | Manchester City  | Manchester City  | Liverpool       | Manchester City |
| LaLiga       | Real Madrid      | Real Madrid      | Atletico Madrid  | Real Madrid     | Barcelona       |
| SerieA       | Napoli           | AC Milan         | Inter Milan      | Juventus        | Juventus        |
| Bundesliga   | Bayern Munich    | Bayern Munich    | Bayern Munich    | Bayern Munich   | Bayern Munich   |
| Ligue1       | Paris S-G        | Paris S-G        | Lille            | Paris S-G       | Paris S-G       |
| Eredivisie   | Ajax             | Ajax             | Ajax             | Ajax            | Ajax            |
| PrimeiraLiga | Benfica          | Porto            | Sporting CP      | Benfica         | Benfica         |


This is the dictionary in dataframe form that I used to add in the winners!

Here is an updated scatterplot with everything so far.

<iframe
  src="Data/WinnerPlot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Framing A Prediction

How do we know there is actually a difference between shots on target and goal metrics for players who won the league and those who did not?
Well, here is a dataframe showing the mean for cases where players won their respective leagues or did not:

| Winners   |    Goals |   Shots On Target |
|:----------|---------:|------------------:|
| False     |  6.60166 |           16.9581 |
| True      | 13.5941  |           29.3663 |

There is a massive gap between the two! This means this is a logical thing to investigate and try to predict!

# Baseline Model

To begin, I took the goals and shots on target columns as predictors, and set the winners column as the target. I trained a KNeighborsClassifier with the number of neighbors at 5. I trained the data and saved the instance for later. 25% of the data would then be testing data. This worked reasonably well, coming back with an accuracy of 0.9435.

Here is a plot showcasing the decision boundary of the model!

[TO BE UPDATED]

It looks alright, but we can do better. Also we must be wary of overfitting. 

# Final Model

Building off my baseline model, I decided to also One Hot Encode the Seasons column to try to improve the accuracy. Styles of play in football can change quickly and significantly, and some years may have star players that score crazy amounts of goals that are hard to replicate. This is a logical metric to OHE. Also implemented a scaler to see if that would improve the accuracy of the model.

The final model consisted of a KNN model pipeline with hyperparameter tuning. It includes feature preprocessing using ColumnTransformer to scale numerical features shots on target and goals and encode the categorical feature season. A Pipeline integrates preprocessing and KNN classification. Hyperparameters are tuned using GridSearchCV with 5-fold cross-validation. The best parameters 'knn_metric': 'euclidean', 'knn_n_neighbors': 11, 'knn_weights': 'uniform' and corresponding accuracy of 0.9545 were retrieved. I wanted to test this to see if there were better ways to weigh distance, calculate distance differently, or get a better number of neighbors. This told me the ideal number of neighbors is 11, and all other hyperparameters were fine as they were. This is slightly higher than the previous model, thus showing an improved accuracy with the same training dataset. 
