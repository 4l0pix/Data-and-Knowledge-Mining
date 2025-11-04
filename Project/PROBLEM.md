# Viticulture and Wine Center
### Problem Specification:
A viticultrure and wine center faces some major problems when trying to optimize water and fertilizers management.
Non-systematic recording and analysis of data leads to:
- Excessive or insufficient irrigation
- Fertilizer waste
- Reduced productivity
- Environmental impact
The company does not have the know-how or equipment to analyze the data it already collects (or could collect) from sensors, times and harvest history.
### Innovation:
The development of a decision support system (DSS) that will propose optimal amounts of water and fertilizers per culture zone, based on previous data and weather forecast.
#
# Data
* Climate(Weather Station):
  1. Temperature
  2. Rainfall
  3. Humidity
  4. Solar Cover
* Ground(Ground Sensors):
  1. Ground Humidity
  2. pH
  3. Noutrients
* Plant Based(iamges):
  1. Stage of Growth
  2. State of Health 
* Historical:
  1. Watering Quantities
  2. Fertilizer Quantities
  3. Crop Yield (kg/m^2)
#
# Storage
* Database: MongoDB
* CSV Entries
* 1-10 GB/Year depending on the frequency of data capture

