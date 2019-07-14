# script name:
# plumber.R

# set API title and description to show up in http://localhost:8000/__swagger__/

#' @apiTitle Run predictions for insurance charges
#' @apiDescription This API takes as insures person data and returns a prediction for the amount of charges 
#' For details on how the model is built, see ?????
#' For further explanations of this plumber function, see https://shirinsplayground.netlify.com/2018/01/plumber/

# load model
# this path would have to be adapted if you would deploy this
load("xgmodel.full.rda") 
load("training.rda")

require("breakDown")
require("breakDown")
require("magrittr")
require("plumber")
require("mlr")
require("xgboost")
#source("requirements.R")

#' Log system time, request method and HTTP user agent of the incoming request
#' @filter logger
function(req){
  cat("System time:", as.character(Sys.time()), "\n",
      "Request method:", req$REQUEST_METHOD, req$PATH_INFO, "\n",
      "HTTP user agent:", req$HTTP_USER_AGENT, "@", req$REMOTE_ADDR, "\n")
  plumber::forward()
}


custom_predict <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)
  response <- pred$data$response
  return(response)
}


# core function follows below:
# define parameters with type and description
# name endpoint
# return output as html/text
# specify 200 (okay) return


#' predict insurance costs prediction
#' @param age:numeric The age of the patient, numeric (min 18 max 64)
#' @param bmi:numeric The body mass indicator, numeric (min 15.96 max 53.13)
#' @param children:numeric The number os children numeric (min 0 max 5)
#' @param smoker:character Does it smoke yes/no
#' @param region:character The region where the people live, character (northeast/northwest/southeast/southwest)
#' @param sex:character Gender male/female
#' @get /predict
#' @html
#' @response 200 Returns Costs prediction from the predictive model;
calculate_prediction <- function(age,sex,bmi,children,smoker,region) {
  
  # make data frame from numeric parameters
  input_data_num <<- data.frame(age=age,
                                sexmale=ifelse(sex=="male",1,0),
                                bmi=as.integer(bmi),
                                children=as.integer(children),
                                smokeryes=ifelse(smoker=="yes",1,0),
                                regionnorthwest=ifelse(region=="northwest",1,0),
                                regionsoutheast=ifelse(region=="southeast",1,0),
                                regionsouthwest=ifelse(region=="southwest",1,0),
                                stringsAsFactors = FALSE)
  
  # and make sure they really are numeric
  input_data_num <<- as.data.frame(t(sapply(input_data_num, as.numeric)))
  
  # validation for parameter
  if (any(is.na(input_data_num))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  if (any(input_data_num < 0) ) {
    res$status <- 400
    res$body <- "Parameters have to be between 0 and 1"
  }
  
  # predict and return result
  pred_rf <<- custom_predict(xgmodel.full, input_data_num)
  paste("----------------\n Test case predicted to be", as.character(pred_rf), "\n----------------\n")
}

#' Explains insurance prediction 
#' @param age:numeric The age of the patient, numeric (min 18 max 64)
#' @param bmi:numeric The body mass indicator, numeric (min 15.96 max 53.13)
#' @param children:numeric The number os children numeric (min 0 max 5)
#' @param smoker:character Does it smoke yes/no
#' @param region:character The region where the people live, character (northeast/northwest/southeast/southwest)
#' @param sex:character Gender male/female
#' @get /explain
#' @png
#' @response 200 Returns Explanation of prediction for the model;
explain_prediction <- function(age,sex,bmi,children,smoker,region) {
  
  # make data frame from numeric parameters
  input_data_num <<- data.frame(age=age,
                                sexmale=ifelse(sex=="male",1,0),
                                bmi=as.integer(bmi),
                                children=as.integer(children),
                                smokeryes=ifelse(smoker=="yes",1,0),
                                regionnorthwest=ifelse(region=="northwest",1,0),
                                regionsoutheast=ifelse(region=="southeast",1,0),
                                regionsouthwest=ifelse(region=="southwest",1,0),
                                stringsAsFactors = FALSE)
  
  # and make sure they really are numeric
  input_data_num <<- as.data.frame(t(sapply(input_data_num, as.numeric)))
  
  # validation for parameter
  if (any(is.na(input_data_num))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  if (any(input_data_num < 0) ) {
    res$status <- 400
    res$body <- "Parameters have to be between 0 and 1"
  }
  

  observation_explain <-
    broken(
      xgmodel.full,
      input_data_num,
      data = training,
      predict.function = custom_predict,
      direction = "up",
      keep_distributions = TRUE
    )
  
  #21print(plot(observation_explain) + title("breakDown plot for prediction"))
  print(plot(observation_explain) )
}
