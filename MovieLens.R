####################################################################
# HarvardX: PH125.9x - Capstone Project                            #
# MovieLens Project                                                #
# https://github.com/HugoAquinoNavarrete/MovieLens                 #
# R code that generates predicted movie ratings and RMSE score     #
# Hugo Aquino                                                      #
# Panama City                                                      #
# August, September 2021                                           #
####################################################################
# v1.0 - Initial version provided by edX HarvardX                  #
# v2.0 - Initial customization                                     #
# v3.0 - Integrating some of the algorithms seen on classes        #
# v4.0 - Integrating recosystem (matrix factorization) algorithm   #
# v5.0 - Improving final table colouring RMSEs results             #
####################################################################

####################################################################
# Dataset section                                                  #
# Create edx set, validation set (final hold-out test set)         # 
# Note: this process could take a couple of minutes                #
####################################################################

####################################################################
# Libraries definition and usage                                   #
####################################################################

# Install libraries with its dependencies 
if(!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages(
  "ggplot2", repos = "http://cran.us.r-project.org",dependencies = TRUE)
if(!require(ggthemes)) install.packages(
  "ggthemes", repos = "http://cran.us.r-project.org",dependencies = TRUE)
if(!require(lubridate)) install.packages(
  "lubridate", repos = "http://cran.us.r-project.org",dependencies = TRUE)
if(!require(corrplot)) install.packages(
  "corrplot", repos = "http://cran.us.r-project.org",dependencies = TRUE)
if(!require(recosystem)) install.packages(
  "recosystem", repos = "http://cran.us.r-project.org",dependencies = TRUE)

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(ggthemes)
library(lubridate)
library(corrplot)
library(recosystem)

####################################################################
# Create edx set, validation set (final hold-out test set)         #
####################################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

####################################################################
# Create dataset for testing and train algorithms from edx dataset #
####################################################################

set.seed(1, sample.kind = "Rounding")

# Split edx dataset into test and train set
testing_index <- createDataPartition(
  y = edx$rating, 
  times = 1, 
  p = 0.1, 
  list = FALSE)

train_set <- edx[-testing_index,]
test_set <- edx[testing_index,]

# To make sure test set contains all movieId, userId from train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

####################################################################
# RMSE function formula                                            #
####################################################################

RMSE <- function(true_rating, predicted_rating){
  sqrt(mean((true_rating - predicted_rating)^2))
}  

####################################################################
# First model - Average rating of all movies across all users      #
####################################################################

# Calculate average rating
mu <- mean(train_set$rating)
mu # Print average rating variable

# Obtain the RMSE of this model
first_model_mu <- RMSE(test_set$rating,mu)

# Store the results on a dataframe
RMSE.results <- data_frame(
  Algorithm = "Model #1 - Average rating movie",
  RMSE = first_model_mu)

####################################################################
# Second model - Movie effect                                      #
####################################################################

# Calculate the bias on rating for movie
movie_avg <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mu)) 

# Calculate predicted values using test_set
predictions_b_movie <- mu + test_set %>%
  left_join(movie_avg, by = "movieId") %>% pull(b_movie)

# If there is a NA when using this effect, replace it with mu
predictions_b_movie <- replace_na(predictions_b_movie, mu)

# Obtain the RMSE of this model
second_model_movie_effect <- RMSE(test_set$rating,predictions_b_movie)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #2 - Movie effect",
    RMSE = second_model_movie_effect ))

# Displays an histogram of the bias movie impact
ggplot(movie_avg, aes(x= b_movie)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color="green") +
  scale_x_continuous(breaks = seq(round(min(movie_avg$b_movie)), 
                                  round(max(movie_avg$b_movie)), by = 0.5)) +
  scale_y_continuous(labels = comma) +
  labs(x = "bias movie", y = "# reviews", caption = "source data: edx dataset") +
  theme_hc() +
  ggtitle("Bias movie effect on rating")

####################################################################
# Third model - User effect                                        #
####################################################################

# Calculate the bias on rating for user
user_avg <- train_set %>% 
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu))

# Calculate predicted values using test_set
predictions_b_user <- mu + test_set %>%
  left_join(user_avg, by = "userId") %>% pull(b_user)

# If there is a NA when using this effect, replace it with mu
predictions_b_user <- replace_na(predictions_b_user, mu)

# Obtain the RMSE of this model
third_model_user_effect <- RMSE(test_set$rating,predictions_b_user)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #3 - User effect",
    RMSE = third_model_user_effect ))

# Displays an histogram of the bias user impact
ggplot(user_avg, aes(x= b_user)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color="green") +
  scale_x_continuous(breaks = seq(round(min(user_avg$b_user)), 
                                  round(max(user_avg$b_user)), by = 0.5)) +
  scale_y_continuous(labels = comma) +
  labs(x = "bias user", y = "# reviews", caption = "source data: edx dataset") +
  theme_hc() +
  ggtitle("Bias user effect on rating")

####################################################################
# Fourth model - Movie plus User effect                            #
####################################################################

# Calculate predicted values using test_set using movie and user effect
predictions_b_movie_b_user <- test_set %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  mutate(predicted = mu + b_movie + b_user) %>% pull(predicted)
  
# If there is a NA when using this effect, replace it with mu
predictions_b_movie_b_user <- replace_na(predictions_b_movie_b_user, mu)

# Obtain the RMSE of this model
fourth_model_movie_user_effect <- RMSE(test_set$rating,predictions_b_movie_b_user)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #4 - Movie plus User effect",
    RMSE = fourth_model_movie_user_effect ))

####################################################################
# Fifth model - Date effect                                        #
####################################################################

# Calculate the bias on rating for date
date_avg <- train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "day")) %>%
  group_by(date) %>%
  summarize(b_date = mean(rating - mu))

# Change format to "yyyy-mm-dd" on "date" variable
date_avg$date <- format(date_avg$date, "%F")

# Add to test_set a new variable "date" converting from epoc
test_set <- test_set %>%
  mutate(date = as.POSIXct(timestamp, origin = "1970-01-01", tz = "GMT"))

# Change format to "yyyy-mm-dd" on "date" variable
test_set$date <- format(test_set$date,"%F")

# Calculate predicted values using test_set
predictions_b_date <- mu + test_set %>%
  left_join(date_avg, by = "date") %>% pull(b_date)

# If there is a NA when using this effect, replace it with mu
predictions_b_date <- replace_na(predictions_b_date, mu)

# Obtain the RMSE of this model
fifth_model_date_effect <- RMSE(test_set$rating,predictions_b_date)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #5 - Date effect",
    RMSE = fifth_model_date_effect ))

# Graph date effect
train_set %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "day")) %>% 
  group_by(date) %>% 
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(date, rating)) + 
  geom_point() + 
  geom_smooth() +
  labs(x = "date", y = "rating", caption = "source data: edx dataset") +
  theme_hc() +
  ggtitle("Date effect on rating")

# Displays an histogram of the date impact
ggplot(date_avg, aes(x= b_date)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color="green") +
  scale_x_continuous(breaks = seq(round(min(date_avg$b_date)), 
                                  round(max(date_avg$b_date)), by = 0.5)) +
  scale_y_continuous(labels = comma) +
  labs(x = "bias date", y = "# reviews", caption = "source data: edx dataset") +
  theme_hc() +
  ggtitle("Bias date effect on rating")

####################################################################
# Sixth model - Movie plus User plus Date effect                   #
####################################################################

# Calculate predicted values using test_set using movie, user and date effect
predictions_b_movie_b_user_b_date <- test_set %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(date_avg, by="date") %>%
  mutate(predicted = mu + b_movie + b_user + b_date) %>% pull(predicted)

# If there is a NA when using this effect, replace it with mu
predictions_b_movie_b_user_b_date <- replace_na(predictions_b_movie_b_user_b_date, mu)

# Obtain the RMSE of this model
sixth_model_movie_user_date_effect <- RMSE(test_set$rating,predictions_b_movie_b_user_b_date)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #6 - Movie plus User plus Date effect",
    RMSE = sixth_model_movie_user_date_effect ))

####################################################################
# Septh model - Genres effect                                      #
####################################################################

# Calculate the bias on rating for genre
genres_avg <- train_set %>%
  group_by(genres) %>%
  summarize(b_genres = mean(rating - mu))

# Calculate predicted values using test_set
predictions_b_genres <- mu + test_set %>%
  left_join(genres_avg, by = "genres") %>% pull(b_genres)

# If there is a NA when using this effect, replace it with mu
predictions_b_genres <- replace_na(predictions_b_genres, mu)

# Obtain the RMSE of this model
septh_model_genres_effect <- RMSE(test_set$rating,predictions_b_genres)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #7 - Genres effect",
    RMSE = septh_model_genres_effect ))

# Displays an histogram of the date impact
ggplot(genres_avg, aes(x= b_genres)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color="green") +
  scale_x_continuous(breaks = seq(round(min(genres_avg$b_genres)), 
                                  round(max(genres_avg$b_genres)), by = 0.5)) +
  labs(x = "bias genres", y = "# reviews", caption = "source data: edx dataset") +
  theme_hc() +
  ggtitle("Bias genres effect on rating")

####################################################################
# Eighth model - Movie plus User plus Date plus Genres effect      #
####################################################################

# Calculate predicted values using test_set using movie, user,
# date and genres effect
predictions_b_movie_b_user_b_date_b_genres <- test_set %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(date_avg, by="date") %>%
  left_join(genres_avg, by="genres") %>%
  mutate(predicted = mu + b_movie + b_user + b_date + b_genres) %>% pull(predicted)

# If there is a NA when using this effect, replace it with mu
predictions_b_movie_b_user_b_date_b_genres <- replace_na(predictions_b_movie_b_user_b_date_b_genres, mu)

# Obtain the RMSE of this model
eighth_model_movie_user_date_genres_effect <- RMSE(test_set$rating,predictions_b_movie_b_user_b_date_b_genres)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #8 - Movie plus User plus Date plus Genres effect",
    RMSE = eighth_model_movie_user_date_genres_effect ))

####################################################################
# Verify correlation                                               #
####################################################################

# Create a dataframe using the "ratings" and "user´s bias prediction"
ratings_results <- cbind(
  test_set$rating,
  predictions_b_user)

# Append the dataframe created with "movie´s bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_movie)

# Append the dataframe created with "date´s bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_date)

# Append the dataframe created with "genres´ bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_genres)

# Append the dataframe created with "movie´s and user´s bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_movie_b_user)

# Append the dataframe created with "movie´s, user´s and date´s bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_movie_b_user_b_date)

# Append the dataframe created with "movie´s, user´s, date´s and genres´ bias prediction"
ratings_results <- cbind(
  ratings_results,
  predictions_b_movie_b_user_b_date_b_genres)

# Change the columns name
colnames(ratings_results) <- c("rating", "user","movie","date","genres","mov_usr","m_u_d","m_u_d_g")

# Calculate and store the correlations
ratings_correlations <- cor(ratings_results)

# Create a correlogram
corrplot(ratings_correlations, 
         type = "lower",
         insig = "blank",
         method = "number",
         title = "Correlation Plot",
         tl.cex = 0.8, 
         cl.cex = 0.65)

####################################################################
# Nineth model - Regularized Movie effect                          #
####################################################################

# Define incremental lambdas values
lambdas <- seq(0, 10, 0.25)

# Function to calculate RMSEs using different lambdas
rmses_reg_movie_effect <- sapply(lambdas, function(lambda){
  
  # Mean
  mu <- mean(train_set$rating)
  
  # Movie effect
  movie_avg <- train_set %>%
    group_by(movieId) %>%
    summarize(reg_movie = sum(rating - mu)/(n()+lambda))
  
  # Predictions: mu + reg_movie
  predictions_reg_movie_effect <- test_set %>% 
    left_join(movie_avg, by = "movieId") %>%
    mutate(predicted = mu + reg_movie) %>% 
    pull(predicted)
    
  return(RMSE(test_set$rating,predictions_reg_movie_effect))
  
})

# Plot relationship between lambdas and RMSEs
qplot(lambdas, 
      rmses_reg_movie_effect,
      xlab = "Lambdas", 
      ylab = "RMSE", 
      colour = "lambdas",
      main = "Lambdas versus RMSE for Regularized Movie effect") +
  geom_vline(
    xintercept = lambdas[which.min(rmses_reg_movie_effect)],
    col = "blue", 
    linetype = "dashed") + 
  geom_hline(
    yintercept = rmses_reg_movie_effect[which.min(rmses_reg_movie_effect)],
    col="blue",
    linetype="dashed")

# Obtain the lambda value to be used on the predictions
lambda_reg_movie_effect <- lambdas[which.min(rmses_reg_movie_effect)]

# Movie effect
reg_movie_avg <- train_set %>%
  group_by(movieId) %>%
  summarize(reg_movie = sum(rating - mu)/(n()+lambda_reg_movie_effect))

# Predictions: mu + reg_movie
predictions_reg_movie_effect <- test_set %>% 
  left_join(reg_movie_avg, by = "movieId") %>%
  mutate(predicted = mu + reg_movie) %>% 
  pull(predicted)

# Obtain the RMSE of this model
nineth_model_reg_movie_effect <- RMSE(test_set$rating,predictions_reg_movie_effect)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #9 - Regularized Movie effect",
    RMSE = nineth_model_reg_movie_effect ))

####################################################################
# Tenth model - Regularized User effect                            #
####################################################################

# Function to calculate RMSEs using different lambdas
rmses_reg_user_effect <- sapply(lambdas, function(lambda){
  
  # Mean
  mu <- mean(train_set$rating)
  
  # User effect
  user_avg <- train_set %>%
    group_by(userId) %>%
    summarize(reg_user = sum(rating - mu)/(n()+lambda))
  
  # Predictions: mu + reg_user
  predictions_reg_user_effect <- test_set %>% 
    left_join(user_avg, by = "userId") %>%
    mutate(predicted = mu + reg_user) %>% 
    pull(predicted)
  
  return(RMSE(test_set$rating,predictions_reg_user_effect))
  
})

# Plot relationship between lambdas and RMSEs
qplot(lambdas, 
      rmses_reg_user_effect,
      xlab = "Lambdas", 
      ylab = "RMSE", 
      colour = "lambdas",
      main = "Lambdas versus RMSE for Regularized User effect") +
  geom_vline(
    xintercept = lambdas[which.min(rmses_reg_user_effect)],
    col = "blue", 
    linetype = "dashed") + 
  geom_hline(
    yintercept = rmses_reg_user_effect[which.min(rmses_reg_user_effect)],
    col="blue",
    linetype="dashed")
  
# Obtain the lambda value to be used on the predictions
lambda_reg_user_effect <- lambdas[which.min(rmses_reg_user_effect)]

# User effect
reg_user_avg <- train_set %>%
  group_by(userId) %>%
  summarize(reg_user = sum(rating - mu)/(n()+lambda_reg_user_effect))

# Predictions: mu + reg_user
predictions_reg_user_effect <- test_set %>% 
  left_join(reg_user_avg, by = "userId") %>%
  mutate(predicted = mu + reg_user) %>% 
  pull(predicted)

# Obtain the RMSE of this model
tenth_model_reg_user_effect <- RMSE(test_set$rating,predictions_reg_user_effect)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #10 - Regularized User effect",
    RMSE = tenth_model_reg_user_effect ))

####################################################################
# Eleventh model - Regularized Movie plus Regularized User effect  #
####################################################################

# Function to calculate RMSEs using different lambdas
rmses_reg_movie_and_reg_user_effect <- sapply(lambdas, function(lambda){
  
  # Mean
  mu <- mean(train_set$rating)
  
  # Movie effect
  movie_avg <- train_set %>%
    group_by(movieId) %>%
    summarize(reg_movie = sum(rating - mu)/(n()+lambda))

  # User effect
  user_avg <- train_set %>%
    left_join(movie_avg, by = "movieId") %>%
    group_by(userId) %>%
    summarize(reg_user = sum(rating - mu - reg_movie)/(n()+lambda))
  
  # Predictions: mu + reg_movie + reg_user
  predictions_reg_movie_and_reg_user_effect <- test_set %>% 
    left_join(movie_avg, by = "movieId") %>%
    left_join(user_avg, by = "userId") %>%
    mutate(predicted = mu + reg_movie + reg_user) %>% 
    pull(predicted)

  return(RMSE(test_set$rating,predictions_reg_movie_and_reg_user_effect))
  
})

# Plot relationship between lambdas and RMSEs
qplot(lambdas, 
      rmses_reg_movie_and_reg_user_effect,
      xlab = "Lambdas", 
      ylab = "RMSE", 
      colour = "lambdas",
      main = "Lambdas versus RMSE for Regularized (Movie plus User) effect")  

# Obtain the lambda value to be used on the predictions
lambda_reg_user_and_reg_movie_effect <- lambdas[which.min(rmses_reg_movie_and_reg_user_effect)]

# Movie effect
reg_movie_avg <- train_set %>%
  group_by(movieId) %>%
  summarize(reg_movie = sum(rating - mu)/(n()+lambda_reg_user_and_reg_movie_effect))

# User effect
reg_user_avg <- train_set %>%
  left_join(reg_movie_avg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(reg_user = sum(rating - mu - reg_movie)/(n()+lambda_reg_user_and_reg_movie_effect))

# Predictions: mu + reg_movie + reg_user
predictions_reg_movie_and_reg_user_effect <- test_set %>% 
  left_join(reg_movie_avg, by = "movieId") %>%
  left_join(reg_user_avg, by = "userId") %>%
  mutate(predicted = mu + reg_movie + reg_user) %>% 
  pull(predicted)

# Obtain the RMSE of this model
eleventh_model_reg_movie_and_reg_user_effect <- RMSE(test_set$rating,predictions_reg_movie_and_reg_user_effect)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(
    Algorithm = "Model #11 - Regularized (Movie plus User) effect",
    RMSE = eleventh_model_reg_movie_and_reg_user_effect ))

####################################################################
# Twelveth model - Matrix factorization                            #
####################################################################

# This process takes from 45 to 60 minutes to run based on the end-user
# computing capabilities 
# The tuning is the task that takes more time to be completed

# Create train set using recosystem´s input format
train_data_recosystem <- with(
  train_set, 
  data_memory(user_index = userId,
              item_index = movieId, 
              rating = rating))

# Create test set using recosystem´s input format
test_data_recosystem <- with(
  test_set,
  data_memory(user_index = userId,
              item_index = movieId, 
              rating = rating))

# Create the model object
matrix_factorization_recosystem <- Reco()

# Use default parameters with exception of nthread from 1 to 6 
opts <- matrix_factorization_recosystem$tune(
  train_data_recosystem, opts = list(dim = c(10L, 20L), 
                                     lrate = c(0.01, 0.1),
                                     costp_l2 = c(0.01, 0.1), 
                                     costq_l2 = c(0.01, 0.1), 
                                     nthread = 6, 
                                     niter = 20))

# Train the algorithm  
matrix_factorization_recosystem$train(
  train_data_recosystem, 
  opts = c(opts$min, nthread = 6, niter = 30))

# Calculate the predicted values  
predictions_matrix_factorization_recosystem <- matrix_factorization_recosystem$predict(test_data_recosystem, out_memory())

# Obtain the RMSE of this model
twelveth_model_matrix_factorization <- RMSE(test_set$rating,predictions_matrix_factorization_recosystem)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(Algorithm = "Model #12 - Matrix factorization using recosystem",
             RMSE = twelveth_model_matrix_factorization ))

####################################################################
# Thirteenth model - Matrix factorization using validation dataset #
####################################################################

# Create validation set using recosystem´s input format
validation_data_recosystem <- with(
  validation, 
  data_memory(user_index = userId,
              item_index = movieId, 
              rating = rating))

# Calculate the predicted values  
predictions_validation_recosystem <- matrix_factorization_recosystem$predict(validation_data_recosystem, out_memory())

# Obtain the RMSE of this model
thirteenth_model_matrix_factorization_validation_recosystem <- RMSE(validation$rating,predictions_validation_recosystem)

# Append the results to the dataframe
RMSE.results <- bind_rows(
  RMSE.results,
  data_frame(Algorithm = "Model #13 - Matrix factorization using recosystem on validation dataset",
             RMSE = thirteenth_model_matrix_factorization_validation_recosystem ))

####################################################################
# Print the dataframe content with the models results              #
####################################################################
options(digits = 6) 

# Coloring the RMSEs obtained with the minimal value on green color
RMSE.results %>% kbl(., 
                     booktabs = T, 
                     caption = "RMSEs obtained") %>% 
  kable_styling(latex_options = "HOLD_position") %>%
  row_spec(0, bold = T) %>% 
  column_spec(
    2, 
    color = "white",
    background = spec_color(
      RMSE.results$RMSE, 
      direction= -1,
      end=0.80),
    popover = paste("RMSE:", RMSE.results$RMSE))

####################################################################
# End of R Capstone script                                         #
####################################################################
