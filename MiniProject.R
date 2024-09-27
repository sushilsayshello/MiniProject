
# Load necessary libraries
library(readr)
library(caret)
library(e1071)  # For Naive Bayes
library(corrplot)
library(ggplot2)
library(GGally)

# Load the dataset
malicious_and_benign_websites1 <- read_csv("Downloads/malicious_and_benign_websites1.csv")

# Handle missing values
dataset <- na.omit(malicious_and_benign_websites1)

# Convert categorical data to factors
dataset$Type <- as.factor(dataset$Type)

# Step 1: Feature Selection
# Calculate correlation matrix
correlation_matrix <- cor(dataset[, sapply(dataset, is.numeric)])

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Select features based on correlation analysis and domain knowledge
selected_features <- c("URL_LENGTH", "NUMBER_SPECIAL_CHARACTERS", 
                       "TCP_CONVERSATION_EXCHANGE", "REMOTE_IPS", "APP_BYTES", "DNS_QUERY_TIMES", "Type")

# Subset the dataset with selected features
dataset <- dataset[selected_features]

# Step 2: Splitting Data into Training and Testing Sets
# Set seed for reproducibility
set.seed(123)
train_index <- sample(1:nrow(dataset), 0.7 * nrow(dataset))
train_data <- dataset[train_index, ]
test_data <- dataset[-train_index, ]

# Step 3: Model Training and Prediction

# Train KNN model
knn_model <- train(Type ~ ., data = train_data, method = 'knn', tuneLength = 5)
# Predict on test data using KNN
knn_predictions <- predict(knn_model, test_data)
# Confusion Matrix for KNN
knn_confusion_matrix <- confusionMatrix(knn_predictions, test_data$Type)


# Train Naive Bayes model with Laplace smoothing
nb_model <- train(Type ~ ., data = train_data, method = 'nb', 
                  trControl = trainControl(method = 'cv', number = 10),
                  tuneGrid = expand.grid(.fL = 1))  # Apply Laplace smoothing

# Identify near-zero variance predictors
nzv <- nearZeroVar(train_data)
# Remove near-zero variance predictors
train_data_nzv <- train_data[, -nzv]
test_data_nzv <- test_data[, -nzv]

# Ensure test data has the same factor levels as training data
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    levels(test_data[[col]]) <- levels(train_data[[col]])
  }
}

nb_model <- train(Type ~ ., data = train_data, method = 'nb', 
                  trControl = trainControl(method = 'cv', number = 10))

nb_predictions <- predict(nb_model, test_data)
nb_confusion_matrix <- confusionMatrix(nb_predictions, test_data$Type)

# Train Naive Bayes model
nb_model <- train(Type ~ ., data = train_data, method = 'nb', trControl = trainControl(method = 'cv', number = 10))
# Predict on test data using Naive Bayes
nb_predictions <- predict(nb_model, test_data)
# Confusion Matrix for Naive Bayes
nb_confusion_matrix <- confusionMatrix(nb_predictions, test_data$Type)
print(nb_confusion_matrix)

# Train Random Forest model
rf_model <- train(Type ~ ., data = train_data, method = 'rf', trControl = trainControl(method = 'cv', number = 10))
# Predict on test data using Random Forest
rf_predictions <- predict(rf_model, test_data)
# Confusion Matrix for Random Forest
rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$Type)
print(rf_confusion_matrix)

# Step 4: Performance Metrics Calculation

# Calculate performance metrics for KNN
knn_accuracy <- knn_confusion_matrix$overall['Accuracy']
knn_precision <- posPredValue(knn_predictions, test_data$Type, positive = "malicious")
knn_recall <- sensitivity(knn_predictions, test_data$Type, positive = "malicious")
knn_f1_score <- (2 * knn_precision * knn_recall) / (knn_precision + knn_recall)

# Calculate performance metrics for Naive Bayes
nb_accuracy <- nb_confusion_matrix$overall['Accuracy']
nb_precision <- posPredValue(nb_predictions, test_data$Type, positive = "malicious")
nb_recall <- sensitivity(nb_predictions, test_data$Type, positive = "malicious")
nb_f1_score <- (2 * nb_precision * nb_recall) / (nb_precision + nb_recall)

# Calculate performance metrics for Random Forest
rf_accuracy <- rf_confusion_matrix$overall['Accuracy']
rf_precision <- posPredValue(rf_predictions, test_data$Type, positive = "malicious")
rf_recall <- sensitivity(rf_predictions, test_data$Type, positive = "malicious")
rf_f1_score <- (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)

# Step 5: Visualization of Performance Metrics
# Create a data frame with performance metrics
metrics_df <- data.frame(
  Model = rep(c("KNN", "Naive Bayes", "Random Forest"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score"), 3),
  Value = c(
    knn_accuracy, knn_precision, knn_recall, knn_f1_score,
    nb_accuracy, nb_precision, nb_recall, nb_f1_score,
    rf_accuracy, rf_precision, rf_recall, rf_f1_score
  )
)

# Plot bar plot of metrics
ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(y = "Value", title = "Performance Metrics for Each Model") +
  theme_minimal()


# Check the levels
levels(knn_predictions)
levels(test_data$Type)

# If there is a mismatch in levels, align them
levels(knn_predictions) <- levels(test_data$Type)
levels(nb_predictions) <- levels(test_data$Type)
levels(rf_predictions) <- levels(test_data$Type)

# Recalculate performance metrics after alignment
knn_confusion_matrix <- confusionMatrix(knn_predictions, test_data$Type)
nb_confusion_matrix <- confusionMatrix(nb_predictions, test_data$Type)
rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$Type)

# Extract and print metrics
# For KNN
knn_accuracy <- knn_confusion_matrix$overall['Accuracy']
knn_precision <- knn_confusion_matrix$byClass["Pos Pred Value"]
knn_recall <- knn_confusion_matrix$byClass["Sensitivity"]
knn_f1_score <- ifelse(!is.na(knn_precision) & !is.na(knn_recall),
                       2 * (knn_precision * knn_recall) / (knn_precision + knn_recall),
                       NA)

print(paste("KNN Accuracy:", knn_accuracy))
print(paste("KNN Precision:", knn_precision))
print(paste("KNN Recall:", knn_recall))
print(paste("KNN F1 Score:", knn_f1_score))

# Check the distribution of the predictions
table(knn_predictions)
table(nb_predictions)
table(rf_predictions)

# Check the levels
levels(knn_predictions)
levels(test_data$Type)

# If there is a mismatch in levels, align them
levels(knn_predictions) <- levels(test_data$Type)
levels(nb_predictions) <- levels(test_data$Type)
levels(rf_predictions) <- levels(test_data$Type)

# Step 5: Recalculate Performance Metrics After Alignment

# Confusion Matrix for KNN
knn_confusion_matrix <- confusionMatrix(knn_predictions, test_data$Type)
# Extract and print metrics for KNN
knn_accuracy <- knn_confusion_matrix$overall['Accuracy']
knn_precision <- knn_confusion_matrix$byClass["Pos Pred Value"]
knn_recall <- knn_confusion_matrix$byClass["Sensitivity"]
knn_f1_score <- ifelse(!is.na(knn_precision) & !is.na(knn_recall),
                       2 * (knn_precision * knn_recall) / (knn_precision + knn_recall),
                       NA)

print(paste("KNN Accuracy:", knn_accuracy))
print(paste("KNN Precision:", knn_precision))
print(paste("KNN Recall:", knn_recall))
print(paste("KNN F1 Score:", knn_f1_score))

# Confusion Matrix for Naive Bayes
nb_confusion_matrix <- confusionMatrix(nb_predictions, test_data$Type)
# Extract and print metrics for Naive Bayes
nb_accuracy <- nb_confusion_matrix$overall['Accuracy']
nb_precision <- nb_confusion_matrix$byClass["Pos Pred Value"]
nb_recall <- nb_confusion_matrix$byClass["Sensitivity"]
nb_f1_score <- ifelse(!is.na(nb_precision) & !is.na(nb_recall),
                      2 * (nb_precision * nb_recall) / (nb_precision + nb_recall),
                      NA)

print(paste("Naive Bayes Accuracy:", nb_accuracy))
print(paste("Naive Bayes Precision:", nb_precision))
print(paste("Naive Bayes Recall:", nb_recall))
print(paste("Naive Bayes F1 Score:", nb_f1_score))

# Confusion Matrix for Random Forest
rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$Type)
# Extract and print metrics for Random Forest
rf_accuracy <- rf_confusion_matrix$overall['Accuracy']
rf_precision <- rf_confusion_matrix$byClass["Pos Pred Value"]
rf_recall <- rf_confusion_matrix$byClass["Sensitivity"]
rf_f1_score <- ifelse(!is.na(rf_precision) & !is.na(rf_recall),
                      2 * (rf_precision * rf_recall) / (rf_precision + rf_recall),
                      NA)

print(paste("Random Forest Accuracy:", rf_accuracy))
print(paste("Random Forest Precision:", rf_precision))
print(paste("Random Forest Recall:", rf_recall))
print(paste("Random Forest F1 Score:", rf_f1_score))

# Step 6: Visualization of Performance Metrics
# Create a data frame with performance metrics
metrics_df <- data.frame(
  Model = rep(c("KNN", "Naive Bayes", "Random Forest"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score"), 3),
  Value = c(
    knn_accuracy, knn_precision, knn_recall, knn_f1_score,
    nb_accuracy, nb_precision, nb_recall, nb_f1_score,
    rf_accuracy, rf_precision, rf_recall, rf_f1_score
  )
)

# Plot bar plot of metrics
ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(y = "Value", title = "Performance Metrics for Each Model") +
  theme_minimal()

# Confusion Matrix Heatmap for Random Forest
library(reshape2)
library(ggplot2)

# Convert confusion matrix to a data frame for visualization
rf_cm_table <- as.data.frame(rf_confusion_matrix$table)
colnames(rf_cm_table) <- c("Reference", "Prediction", "Freq")

# Plot heatmap
ggplot(rf_cm_table, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix Heatmap for Random Forest", fill = "Count")


# Plot bar plot of metrics
ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(y = "Value", title = "Performance Metrics for Each Model") +
  theme_minimal()

# Create a data frame with performance metrics
metrics_df <- data.frame(
  Model = rep(c("KNN", "Naive Bayes", "Random Forest"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score"), 3),
  Value = c(
    knn_accuracy, knn_precision, knn_recall, knn_f1_score,
    nb_accuracy, nb_precision, nb_recall, nb_f1_score,
    rf_accuracy, rf_precision, rf_recall, rf_f1_score
  )
)

# Plot bar plot of metrics
ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(y = "Value", title = "Performance Metrics for Each Model") +
  theme_minimal()

# Install and load fmsb package if not already installed
if (!require("fmsb")) install.packages("fmsb")
library(fmsb)

# Combine the metrics into a data frame suitable for radar chart
metrics_df <- data.frame(
  row.names = c("Max", "Min", "KNN", "Naive Bayes", "Random Forest"),
  Accuracy = c(1, 0, knn_accuracy, nb_accuracy, rf_accuracy),
  Precision = c(1, 0, knn_precision, nb_precision, rf_precision),
  Recall = c(1, 0, knn_recall, nb_recall, rf_recall),
  F1_Score = c(1, 0, knn_f1_score, nb_f1_score, rf_f1_score)
)

# Display the data frame
print(metrics_df)

# Create the radar chart
radarchart(
  metrics_df,
  axistype = 1,
  pcol = c(NA, NA, "blue", "green", "red"), # Colors for each model
  pfcol = c(NA, NA, rgb(0, 0, 255, alpha = 0.2), rgb(0, 255, 0, alpha = 0.2), rgb(255, 0, 0, alpha = 0.2)),
  plwd = 2, # Line width
  plty = 1, # Line type
  title = "Model Performance Metrics Radar Chart"
)

# Add a legend
legend("topright", legend = c("KNN", "Naive Bayes", "Random Forest"),
       col = c("blue", "green", "red"), lty = 1, lwd = 2, bty = "n")

# Install and load knitr package if not already installed
if (!require("knitr")) install.packages("knitr")
library(knitr)

# Create a data frame with the results
results_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  KNN = c(knn_accuracy, knn_precision, knn_recall, knn_f1_score),
  Naive_Bayes = c(nb_accuracy, nb_precision, nb_recall, nb_f1_score),
  Random_Forest = c(rf_accuracy, rf_precision, rf_recall, rf_f1_score)
)

# Generate the table using kable
kable(results_df, format = "markdown", digits = 3, caption = "Performance Metrics for Each Model")


