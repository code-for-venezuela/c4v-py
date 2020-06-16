# brat-test

This repo contains the first iterations of Code For Venezuela's Named-Entity Recognition for public services outages in Venezuela. The goal of this project is to identify the following entities and attributes in tweets that utilize the public services-related tweets. E.g. #SinLuz, #SinAgua, #SinGas, #SinGasolina, etc...

## Data and Notebooks

- First iteration of annotations can be found at [./brat-v1.3_Crunchy_Frog/data/first-iter](https://github.com/dieko95/brat-test/tree/master/brat-v1.3_Crunchy_Frog/data/first-iter)
- Jupyter Notebooks with descriptive analytics about the annotations can be found at [./data_analysis](https://github.com/dieko95/brat-test/tree/master/data_analysis)

## Steps to annotate

### Change to Brat's directory
`cd brat-test/brat-v1.3_Crunchy_Frog`

### Run Brat's standalone server
`python2 standalone.py`

### Login into brat
Once inside brat, click login on the right top corner of the browser.
- User: admin
- Pass: admin

### Annotate
Click on Collections and select the file to annotate

## Entities and attributes

Entities as bullet points and attributes as sub-bullet points:

- water
  - with-service
  - without-service
- electricity
  - with-service
  - without-service
- gas
  - with-service
  - without-service
- gasoline
  - with-service
  - without-service
- social-report
- circumstantial-information
  - duration
  - location
  - time
  - reason
- twitter-account
  - politician
  - utility-company
  - news-company
  - other
