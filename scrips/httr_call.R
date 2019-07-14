library(httr)
# Data.gov Utility Rates API
request <- GET("http://localhost:8080/predict?age=18&sex=female&bmi=50&children=0&smoker=yes&region=northwest")
content(request)
