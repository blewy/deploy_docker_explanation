FROM rocker/verse:latest
MAINTAINER me

COPY ./requirements.R /tmp/requirements.R 
RUN Rscript /tmp/requirements.R

## uncomment to include shiny server
# #RUN export ADD=shiny && bash /etc/cont-init.d/add

# create an R user
ENV USER rstudio

## Copy your working files over
COPY .  /home/$USER/API

## Setting work directory
WORKDIR /home/$USER/API
EXPOSE 8000
CMD ["Rscript", "-e", "pr <- plumber::plumb('plumber.R'); pr$run(host='0.0.0.0', port=8000)"]
