library(plumber)
setwd("./insurance_api")
#source("requirements.R")
r <- plumb("plumber.R")
r$run(port=8000)
