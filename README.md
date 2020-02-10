# Air-Pressure-System-Failures-Prediction-in-Scania-Trucks
1.INTRODUCTION

The whole dataset includes two parts. The training set contains 60,000 examples in total in which 59,000 belong to the negative class and 1,000 belong to the positive class. The test set contains 16,000 examples. There are 171 attributes per record. Both the training and test set include the class label.
The attribute names of the data have been anonymized for proprietary reasons. The attribute values of these features are numerical counters. The attributes are as follows: class, then anonymized operational data. The operational data have an identifier and a bin id, like "Identifier_Bin". In total there are 171 attributes, of which 7 are histogram variables. Missing values are denoted by "na".
The total cost of a prediction model is the sum of Cost_1 multiplied by the number of instances with type 1 failure and Cost_2 with the number of instances with type 2 failure, resulting in a Total_Cost. In this case Cost_1 refers to the cost that an unnecessary check needs to be done by a mechanic at a workshop, while Cost_2 refer to the cost of missing a faulty truck, which may cause a breakdown. Cost_1 = 10 and Cost_2 = 500, as in:
      Total_cost= Cost_1*No_Instances + Cost_2*No_Instances
From the above problem statement, we could observe that we have to reduce False Positives and False Negatives. More importantly we have to reduce False Negatives, since cost incurred due to False Negatives is 50 times higher than the False Positives.
Because we have to reduce False Negatives and False Positives, so clearly we can use precision and recall as performance metric. But here is a simpler metric which takes into account both precision and recall, and therefore, we can aim to maximize this number to make your model better. This metric is known as F1-score, which is simply the harmonic mean of precision and recall.

2.EXPERIMENT METHODOLOGY

A.	Data Preprocessing

The project starts with data preprocessing. From the target distribution graph, we can conclude that the given data is highly imbalanced. In the training set, there are 59,000 samples belong to the negative class and 1,000 samples belong to the positive class So we need to balance the data later.
Also, the attribute values of class column are ‘pos’ and ‘neg’, we replace these with number ‘1’ and ‘0’.
Then we come to the missing value. In the given data, missing values are represented as ‘na’, we will turn ‘na’ to ‘NaN’ format to process the data. We can see there are lots of missing values in the data, so we cannot remove the rows as it will leads to the great amount of information loss.
To avoid the information loss, we will implement below two ideas to deal with the missing values. The first idea is to remove columns which contain more than 75% of ‘NaN’ values, and then, we impute the remaining missing values with mean and median value of that feature. With the help of these two methods, now we have 160 features left.
Now we have two dataframes, one with the mean imputation, and the other with the median imputation. Next we will use SMOTE to balance these two dataframes separately. SMOTE is an oversampling method. It works by creating synthetic samples from the minor class instead of creating copies. The algorithm selects two or more similar instances (using a distance measure) and perturbing an instance one attribute at a time by a random amount within the difference to the neighboring instances.

B.	Random Model

Before we come to the part of model building. At first, we will build a random model as a benchmark for all our models. The output is the confusion matrix and F-1 score of the random model, we can see the F-1 score is 0.045 and the cost is 171,980. Then, we will come to our model building. If the following models we build next has lower matrix than random model, we can ignore that model.

C.	Data Mining Algorithm

I have built two models with two different dataframes. They are random forest model with mean value, logistic regression model with mean value, random forest model with median value, and logistic regression model with median value. In each model, I split the training dataset into train and test part, I set 70% as training partition and 30% as test patition. And I used the original test dataset to evaluate the performance of each model. When evaluating, I introduced F1-score as a measure to check the performance of the models.
Random forest is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forest corrects for decision trees' habit of overfitting to their training set.
Logistic regression is a statistical method for analysing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall). In this case, we need to pay more attention to False Negatives, since cost incurred due to False Negatives is 50 times higher than the False Positives. Which means, we need to focus more on recall.
In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F-1 score is the harmonic mean of the precision and recall, where an F-1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

3.RESULTS

As mentioned above, we have built four models, here comes the precision and recall curve of these four models.
From the results, we can see that random forest model with mean value has the highest decision threshold, so we may conclude this model performed best so far. For more details, let’s check the F-1 score and cost.
The lats output shows the F1-score and total cost of each model, from this figure, we can see random forest model with mean value has the highest F-1 score, but the total cost of which is high, too. And the random forest model with median value has the lowest cost. So in order to make the decision, we may need to do some further work like building other models or explore the random forest model with mean value for a better performance. For now, we will choose random forest model with mean value as our final model.

