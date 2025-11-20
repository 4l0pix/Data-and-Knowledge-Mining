import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class VineyardDataGenerator:
    def __init__(self, config_path='vineyard_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        # start two years back
        today = datetime.now()
        self.start_date = today - timedelta(days=730)
        self.historical_days = 730  # two-year span
        self.forecast_days = 30  # 30-day forecast
        
        # seasonal profile for crete
        self.seasonal_profiles = {
            'winter': {'months': [12, 1, 2], 'temp_range': (8, 18), 'humidity_range': (65, 85)},
            'spring': {'months': [3, 4, 5], 'temp_range': (12, 26), 'humidity_range': (55, 75)},
            'summer': {'months': [6, 7, 8], 'temp_range': (20, 36), 'humidity_range': (45, 65)},
            'autumn': {'months': [9, 10, 11], 'temp_range': (15, 28), 'humidity_range': (55, 75)}
        }
        
        # microclimate tweak per zone
        self.zone_microclimate = {}
        for zone_id in self.config['sensors'].keys():
            self.zone_microclimate[zone_id] = {
                'temp_offset': np.random.uniform(-1.5, 1.5),  # zone temperature offset
                'humidity_offset': np.random.uniform(-8, 8)     # zone humidity offset
            }
    
    def _get_seasonal_profile(self, month):
        """Get seasonal climate profile for a given month"""
        for season, profile in self.seasonal_profiles.items():
            if month in profile['months']:
                return profile
        return self.seasonal_profiles['spring']  # default fallback
        
    def generate_all_data(self):
        dates = pd.date_range(self.start_date, periods=self.historical_days + self.forecast_days, freq='D')
        # drop time part
        dates = dates.date
        weather_data = self._generate_weather(dates)
        sensor_data = self._generate_sensor_data(dates, weather_data)
        plant_data = self._generate_plant_data(dates, sensor_data)
        intervention_data = self._generate_interventions(dates)
        
        weather_data.to_csv('weather_data.csv', index=False)
        sensor_data.to_csv('sensor_data.csv', index=False)
        plant_data.to_csv('plant_data.csv', index=False)
        intervention_data.to_csv('intervention_data.csv', index=False)
        print(f"Generated data for {len(dates)} days")
        
    def _generate_weather(self, dates):
        n = len(dates)
        temp = np.zeros(n)
        humidity = np.zeros(n)
        
        # apply seasonal profile
        for i, date in enumerate(dates):
            month = date.month
            profile = self._get_seasonal_profile(month)
            
            # pick seasonal temp
            temp_min, temp_max = profile['temp_range']
            daily_temp_base = np.random.uniform(temp_min, temp_max)
            temp[i] = daily_temp_base + np.random.normal(0, 2)  # add daily noise
            
            # pick seasonal humidity
            hum_min, hum_max = profile['humidity_range']
            daily_hum_base = np.random.uniform(hum_min, hum_max)
            humidity[i] = daily_hum_base + np.random.normal(0, 5)
        
        humidity = np.clip(humidity, 30, 95)
        
        base_rain = 2 * (1 + np.sin(2 * np.pi * (np.arange(n) + 90) / 365))
        rainfall = np.maximum(0, base_rain + np.random.exponential(1.5, n))
        rainfall[np.random.rand(n) > 0.3] = 0  # enforce dry days
        
        solar = 200 + 150 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 20, n)
        solar = np.clip(solar, 50, 400)
        
        return pd.DataFrame({
            'date': dates,
            'temperature': temp,
            'rainfall': rainfall,
            'humidity': humidity,
            'solar_radiation': solar,
            'is_forecast': [i >= self.historical_days for i in range(n)]
        })
    
    def _generate_sensor_data(self, dates, weather_data):
        rows = []
        for zone_id, sensors in self.config['sensors'].items():
            # fetch zone offset
            zone_temp_offset = self.zone_microclimate[zone_id]['temp_offset']
            zone_humidity_offset = self.zone_microclimate[zone_id]['humidity_offset']
            
            for sensor in sensors:
                # give sensor unique state
                moisture = np.zeros(len(dates))
                moisture[0] = np.random.uniform(15, 35)  # wide start moisture
                
                # add sensor variance
                moisture_retention = np.random.uniform(0.6, 1.4)  # soil variance
                drainage_rate = np.random.uniform(0.03, 0.20)     # drainage variance
                base_moisture_offset = np.random.uniform(-5, 5)   # sensor moisture bias
                
                # sensor microclimate tweak
                sensor_temp_micro = np.random.uniform(-0.5, 0.5)  # small temp tweak
                sensor_humidity_micro = np.random.uniform(-3, 3)   # humidity tweak
                
                for i in range(1, len(dates)):
                    # read base weather
                    base_temp = weather_data.loc[i, 'temperature']
                    base_humidity = weather_data.loc[i, 'humidity']
                    
                    # apply zone and sensor offsets
                    sensor_temp = base_temp + zone_temp_offset + sensor_temp_micro
                    sensor_humidity = base_humidity + zone_humidity_offset + sensor_humidity_micro
                    
                    # moisture follows climate
                    # heat dries soil
                    # humidity slows loss
                    temp_effect = (sensor_temp - 20) * 0.15  # heat loss factor
                    humidity_effect = (sensor_humidity - 60) * 0.05  # humidity buffer factor
                    
                    rain_effect = weather_data.loc[i, 'rainfall'] * 0.3 * moisture_retention
                    evap_effect = temp_effect - humidity_effect + weather_data.loc[i, 'solar_radiation'] * 0.01
                    drainage = moisture[i-1] * drainage_rate
                    
                    # boost depletion in forecast
                    future_factor = 1.0 + (0.3 * (i - self.historical_days) / self.forecast_days) if i >= self.historical_days else 1.0
                    
                    moisture[i] = moisture[i-1] + rain_effect - (evap_effect * future_factor) - drainage + base_moisture_offset * 0.02 + np.random.normal(0, 1.5)
                    moisture[i] = np.clip(moisture[i], 10, 40)
                
                # sensor ph baseline
                pH_base = np.random.uniform(5.8, 6.6)
                pH = pH_base + 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.15, len(dates))
                pH = np.clip(pH, 5.5, 7.0)
                
                # sensor nutrient baseline
                N = np.zeros(len(dates))
                N[0] = np.random.uniform(20, 45)  # wide start nitrogen
                N_depletion = np.random.uniform(0.010, 0.035)  # varied depletion
                N_base_offset = np.random.uniform(-3, 3)  # nitrogen bias
                
                for i in range(1, len(dates)):
                    # boost depletion for forecast
                    future_factor = 1.0 + (0.5 * (i - self.historical_days) / self.forecast_days) if i >= self.historical_days else 1.0
                    
                    N[i] = N[i-1] - (N_depletion * future_factor) + N_base_offset * 0.03 + np.random.normal(0, 1.2)
                    if i % 90 == 0 and i < self.historical_days:  # quarterly fertilization history
                        N[i] += np.random.uniform(10, 20)  # variable boost
                    N[i] = np.clip(N[i], 10, 50)
                
                P_base = np.random.uniform(15, 28)
                P = P_base + np.random.normal(0, 3.5, len(dates)) - np.arange(len(dates)) * np.random.uniform(0.002, 0.010)
                P = np.clip(P, 10, 35)
                
                K_base = np.random.uniform(160, 240)
                K = K_base + np.random.normal(0, 18, len(dates)) - np.arange(len(dates)) * np.random.uniform(0.010, 0.035)
                K = np.clip(K, 120, 280)
                
                for i, date in enumerate(dates):
                    # compute sensor climate
                    base_temp = weather_data.loc[i, 'temperature']
                    base_humidity = weather_data.loc[i, 'humidity']
                    
                    # apply zone plus sensor tweak
                    sensor_temp = base_temp + zone_temp_offset + sensor_temp_micro + np.random.normal(0, 0.3)
                    sensor_humidity = base_humidity + zone_humidity_offset + sensor_humidity_micro + np.random.normal(0, 2.0)
                    sensor_humidity = np.clip(sensor_humidity, 30, 95)
                    
                    rows.append({
                        'date': date,
                        'sensor_id': sensor['sensor_id'],
                        'zone_id': zone_id,
                        'ground_moisture': moisture[i],
                        'temperature': sensor_temp,
                        'humidity': sensor_humidity,
                        'pH': pH[i],
                        'nutrient_N': N[i],
                        'nutrient_P': P[i],
                        'nutrient_K': K[i]
                    })
        return pd.DataFrame(rows)
    
    def _generate_plant_data(self, dates, sensor_data):
        rows = []
        for zone_id in self.config['sensors'].keys():
            zone_sensors = sensor_data[sensor_data['zone_id'] == zone_id]
            
            for i, date in enumerate(dates):
                month = date.month
                stage = next((s['stage'] for s in self.config['growth_stages'] if month in s['months']), 'Dormant')
                
                day_moisture = zone_sensors[zone_sensors['date'] == date]['ground_moisture'].mean()
                day_N = zone_sensors[zone_sensors['date'] == date]['nutrient_N'].mean()
                
                health = 0.7 + 0.1 * (day_moisture - 20) / 20 + 0.1 * (day_N - 20) / 20 + np.random.normal(0, 0.05)
                health = np.clip(health, 0.3, 1.0)
                
                rows.append({
                    'date': date,
                    'zone_id': zone_id,
                    'growth_stage': stage,
                    'health_index': health
                })
        return pd.DataFrame(rows)
    
    def _generate_interventions(self, dates):
        rows = []
        for zone_id in self.config['sensors'].keys():
            for i, date in enumerate(dates):
                if i < self.historical_days and i % 14 == 0:  # fortnight irrigation history
                    rows.append({
                        'date': date,
                        'zone_id': zone_id,
                        'water_applied': np.random.uniform(15, 25),
                        'fertilizer_N_applied': 0,
                        'fertilizer_P_applied': 0,
                        'fertilizer_K_applied': 0
                    })
                if i < self.historical_days and i % 90 == 0:  # quarterly nutrient history
                    rows.append({
                        'date': date,
                        'zone_id': zone_id,
                        'water_applied': 0,
                        'fertilizer_N_applied': np.random.uniform(5, 10),
                        'fertilizer_P_applied': np.random.uniform(3, 6),
                        'fertilizer_K_applied': np.random.uniform(8, 12)
                    })
        return pd.DataFrame(rows)

if __name__ == '__main__':
    gen = VineyardDataGenerator()
    gen.generate_all_data()
