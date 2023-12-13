# SolarRadiationForecasting
Code for Master's dissertation: Solar Radiation Forecasting with Deep Learning and Conformal Prediction

## Day Ahead Forecasting

Approach uses Time Series Forecasting \
Experimenting with both TiDE (Time Series Dense Encoder) and a MLP \
Features include historical solar radiation, weather and time data

### Conformal Prediction

CopulaCPTS algorithm used to measure uncertainty of day-ahead forecasts \
Algorithm adjusted for solar radiation forecasting \ 
Code in DayAheadForecasting/CopulaCPTS folder \

# Sky Image Regression

Approach predicts solar radiation levels based on images of the sky \
Fine-tuned ResNet-18 and Vision Transformer on sky images \ 

