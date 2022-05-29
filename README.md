# Rossmann Pharmaceuticals Sales Prediction


## Objective


The finance team at Rossmann Pharmaceuticals want to forecast sales in all their stores across several cities six weeks ahead of time. The want to use the data they have to be able to predict the sales ahead of time. The job is to build and serve an end-to-end product that delivers this prediction to analysts in the finance team.

## Approach


The project consists of the following phases
* Exploratory Data Analysis
* Sales Prediction
    - Machine Learning Approach using RandomForestRegressor
    - Deep Learning Approach using LSTM
* A web interface to serve model


## Data Fields


* Id - an Id that represents a (Store, Date) duple within the test set  
* Store - a unique Id for each store  
* Sales - the turnover for any given day (this is what you are predicting)  
* Customers - the number of customers on a given day  
* Open - an indicator for whether the store was open: 0 = closed, 1 = open  
* StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None  
* SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools  
* StoreType - differentiates between 4 different store models: a, b, c, d  
* Assortment - describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here  
* CompetitionDistance - distance in meters to the nearest competitor store  
* CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened  
* Promo - indicates whether a store is running a promo on that day  
* Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating  
* Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2  
* PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store  


## Installation

```
git clone https://github.com/teddyk251/pharma_sales_prediction.git
cd pharma_sales_prediction
python setup.py install
```
