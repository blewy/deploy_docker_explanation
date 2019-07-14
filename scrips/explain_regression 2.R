#library(MASS)
#library(mlbench)
library(tidyverse)
library(caret)
#library(mlr)
#devtools::install_github("laresbernardo/lares")
library(lares)
library(gbm)
library(mlr)

#https://www.r-bloggers.com/dalex-and-h2o-machine-learning-model-interpretability-and-feature-explanation/

# Get data
insurance.data <- read.csv("./data/insurance.csv")
View(insurance.data)
str(insurance.data)
summarizeColumns(insurance.data)
summarizeLevels(insurance.data, cols = NULL)

#correct Names
names(insurance.data)  <- make.names(names(insurance.data))
#View(insurance.data)
id.data <- insurance.data$competitorname


# Basic density
ggplot(insurance.data,
       aes(
         x = charges,
         y = ..density..,
         color = sex,
         fill = sex
       ),
       legend = TRUE) +
  labs(title = "Cost Distribuition") +
  geom_density(alpha = 0.2)  + geom_vline(
    data = insurance.data,
    aes(xintercept = mean(charges), colour = sex),
    linetype = "dashed",
    size = 0.5
  )  + theme_minimal()

names(insurance.data)

# Basic density
ggplot(insurance.data,
       aes(
         x = charges,
         y = ..density..,
         color = smoker,
         fill = smoker
       ),
       legend = TRUE) +
  labs(title = "Cost Distribuition") +
  geom_density(alpha = 0.2)  + geom_vline(
    data = insurance.data,
    aes(xintercept = mean(charges), colour = smoker),
    linetype = "dashed",
    size = 0.5
  )  + theme_minimal()


# plot quantile costs
plot.data <- insurance.data
plot.data$cut <-
  cut(
    plot.data$charges,
    breaks = quantile(
      plot.data$charges,
      probs = seq(0, 1, by = 0.05),
      na.rm = TRUE
    ),
    include.lowest = TRUE
  )
plot.data.long <- plot.data %>% group_by(cut) %>%
  summarise(costs = sum(charges),
            proportion = costs / sum(plot.data$charges)) %>%
  gather(key, value, costs)
ggplot(plot.data.long , aes(x = cut, y = proportion)) + geom_col(alpha =
                                                                   0.7,
                                                                 aes(fill = key, color = key),
                                                                 position = "dodge") + labs(
                                                                   title = "Quantile Plot",
                                                                   subtitle = "Comparing observed vs predicted values",
                                                                   x = "Quantile (obs. Values)",
                                                                   y = "Sum of Loss"
                                                                 ) + theme_light() + theme(axis.text.x  = element_text(
                                                                   angle = 45,
                                                                   vjust = 0.5,
                                                                   size = 6
                                                                 ))


insurance.data.model <-
  as.data.frame(model.matrix( ~ . - 1, insurance.data))
insurance.data.model$sexfemale <- NULL
#View(insurance.data.model)

#Split Data
n = nrow(insurance.data.model)
set.seed(1951)
trainIndex = sample(1:n, size = round(0.7 * n), replace = FALSE)
id <- 1:n
training = insurance.data.model[trainIndex , ]
id_training <- id[trainIndex]
testing = insurance.data.model[-trainIndex , ]
id_testing <- id[-trainIndex]


#look at variable classes
split(names(insurance.data), sapply(insurance.data, function(x) {
  class(x)
}))
#splitting the data based on class
split(names(insurance.data.model),
      sapply(insurance.data.model, function(x) {
        class(x)
      }))
#View(insurance.data.model)

#create a task
trainTask <- makeRegrTask(data =  training, target = "charges")
save(training, file = "./insurance_api/training.rda")

#create a task
testnTask <- makeRegrTask(data =  testing, target = "charges")

#create a task
alldataTask <- makeRegrTask(data =  insurance.data.model, target = "charges")

#Xgboost
#load xgboost
set.seed(1001)
getParamSet("regr.xgboost")

#make learner with inital parameters
xg_set <- makeLearner("regr.xgboost", predict.type = "response")
xg_set$par.vals <- list(
  objective = "reg:gamma",
  eval_metric = "rmse",
  nrounds = 500,
  early_stopping_rounds = 20,
  print_every_n = 10
  
)

#define parameters for tuning
xg_ps <- makeParamSet(
  #makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth", lower = 3, upper = 20),
  makeNumericParam("lambda", lower = 0.55, upper = 0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight", lower = 1, upper = 5),
  makeNumericParam("colsample_bytree", lower = 0.2, upper = 0.8)
)

#define search function
rancontrol <- makeTuneControlRandom(maxit = 5L) #do 100 iterations

#5 fold cross validation
set_cv <- makeResampleDesc("CV", iters = 2L)

#tune parameters
xg_tune <- tuneParams(
  learner = xg_set,
  task = trainTask,
  resampling = set_cv,
  measures = rmse,
  par.set = xg_ps,
  control = rancontrol
)

data = generateHyperParsEffectData(
  xg_tune ,
  include.diagnostics = FALSE,
  trafo = FALSE,
  partial.dep = TRUE
)

#cross validation results
data$data

#set parameters
xg_new <- setHyperPars(learner = xg_set, par.vals = xg_tune$x)

#train model with the best parameters
xgmodel <- train(xg_new, trainTask)

#predict on test data
predict.xg <- predict(xgmodel, testnTask)
predict.xg$data$error<-abs(predict.xg$data$truth-predict.xg$data$response)
str(predict.xg$data)
ggplot(data = predict.xg$data, aes(x = truth, y = response , color= as.factor(testing$sexmale), size=predict.xg$data$error)) + geom_point() +  geom_smooth()


#train model with the best parameters
xgmodel.full <- train(xg_new, alldataTask)
save(xgmodel.full, file = "./insurance_api/xgmodel.full.rda")



# lares::mplot_lineal(tag = testing$charges,
#                    score = testing$predictions,
#                    subtitle = "Insurance Costs",
#                    model_name = "gbm_model")
#
# lares::mplot_cuts_error(tag = testing$charges,
#                     score = testing$predictions,
#                     title = "Insurance Costs",
#                     model_name = "gbm_model")
#
# lares::mplot_density(tag = testing$charges,
#                      score = testing$predictions,
#                      subtitle = "Insurance Costs",
#                      model_name = "gbm_model")

# lares::mplot_full(tag = predict.xg$data$truth,
#                   score = predict.xg$data$response,
#                   splits = 10,
#                   subtitle = "Insurance Costs",
#                   model_name = "xgb_model",
#                   save = F)

#lares::mplot_splits(tag = testing$charges,
#                    score = testing$predictions,
#                    split = 8)


#---------------------- BreakDown package ----------------------------
#https://pbiecek.github.io/breakDown/articles/break_caret.html
#-----------------------------------------------------------------------

library(breakDown)

#"Help function for MLR predictions"
custom_predict <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)
  response <- pred$data$response
  return(response)
}


observation_explain <-
  broken(
    xgmodel,
    testing[2, ],
    data = training[,-ncol(training)],
    predict.function = custom_predict,
    direction = "down",
    keep_distributions = TRUE
  )
observation_explain

plot(observation_explain,top_features = 10) + ggtitle("breakDown plot")

