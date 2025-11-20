import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spatial_interpolation import SpatialInterpolator

class VineyardDSS:
    def __init__(self, config_path='vineyard_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.interpolator = SpatialInterpolator(config_path)
    
    def calculate_et(self, temp, solar, humidity):
        """Simplified Hargreaves ET calculation"""
        temp_max = temp + 5
        temp_min = temp - 5
        et = 0.0023 * (temp + 17.8) * (temp_max - temp_min)**0.5 * solar / 41.868
        et *= (1 - humidity / 200)  # humidity trim
        return max(0, et)
    
    def calculate_water_prescription(self, target_date_str):
        """Calculate water needs for target date"""
        sensor_data = pd.read_csv('sensor_data.csv')
        weather_data = pd.read_csv('weather_data.csv')
        plant_data = pd.read_csv('plant_data.csv')
        intervention_data = pd.read_csv('intervention_data.csv')
        
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        today = sensor_data['date'].max()
        today_dt = datetime.strptime(today, '%Y-%m-%d')
        
        # compute day gap
        days_ahead = (target_date - today_dt).days
        
        forecast_data = weather_data[weather_data['date'] > today]
        forecast_data = forecast_data[forecast_data['date'] <= target_date_str]
        
        cumulative_et = 0
        cumulative_rain = 0
        for _, row in forecast_data.iterrows():
            cumulative_et += self.calculate_et(row['temperature'], row['solar_radiation'], row['humidity'])
            cumulative_rain += row['rainfall']
        
        # boost needs with time
        time_multiplier = 1.0 + max(0, days_ahead * 0.05)  # five percent daily
        
        effective_rain = cumulative_rain * 0.7  # seventy percent efficiency
        
        # compute sensor water
        water_needs = {}
        sensor_needs = {}
        current_sensor = sensor_data[sensor_data['date'] == target_date_str]
        current_plant = plant_data[plant_data['date'] == target_date_str]
        
        for zone_id in self.config['sensors'].keys():
            zone_sensors = current_sensor[current_sensor['zone_id'] == zone_id]
            zone_plant = current_plant[current_plant['zone_id'] == zone_id]
            
            growth_stage = zone_plant['growth_stage'].values[0] if len(zone_plant) > 0 else 'Dormant'
            
            stage_info = next((s for s in self.config['growth_stages'] if s['stage'] == growth_stage), 
                            self.config['growth_stages'][0])
            water_factor = stage_info['water_factor']
            
            # check recent irrigation
            recent_irrigation = intervention_data[
                (intervention_data['zone_id'] == zone_id) & 
                (intervention_data['date'] > (today_dt - timedelta(days=7)).strftime('%Y-%m-%d'))
            ]
            recent_water = recent_irrigation['water_applied'].sum()
            residual_water = recent_water * 0.3  # leave thirty percent
            
            # loop sensor entries
            zone_water_needs = []
            for _, sensor_row in zone_sensors.iterrows():
                sensor_id = sensor_row['sensor_id']
                sensor_moisture = sensor_row['ground_moisture']
                
                water_deficit = (cumulative_et * water_factor * time_multiplier) - effective_rain - (sensor_moisture - 20) - residual_water
                sensor_water_need = max(0, water_deficit)
                
                sensor_needs[sensor_id] = {
                    'water_mm': round(sensor_water_need, 2),
                    'zone_id': zone_id,
                    'current_moisture': round(sensor_moisture, 2)
                }
                zone_water_needs.append(sensor_water_need)
            
            # keep zone average
            water_needs[zone_id] = round(np.mean(zone_water_needs), 2) if zone_water_needs else 0
        
        return water_needs, sensor_needs
    
    def calculate_fertilizer_prescription(self, target_date_str):
        """Calculate fertilizer needs for target date"""
        sensor_data = pd.read_csv('sensor_data.csv')
        plant_data = pd.read_csv('plant_data.csv')
        
        today = sensor_data['date'].max()
        today_dt = datetime.strptime(today, '%Y-%m-%d')
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        # compute day gap
        days_ahead = (target_date - today_dt).days
        time_multiplier = 1.0 + max(0, days_ahead * 0.06)  # six percent daily
        
        current_sensor = sensor_data[sensor_data['date'] == target_date_str]
        current_plant = plant_data[plant_data['date'] == target_date_str]
        
        fertilizer_needs = {}
        sensor_fertilizer_needs = {}
        
        for zone_id in self.config['sensors'].keys():
            zone_sensors = current_sensor[current_sensor['zone_id'] == zone_id]
            zone_plant = current_plant[current_plant['zone_id'] == zone_id]
            
            growth_stage = zone_plant['growth_stage'].values[0] if len(zone_plant) > 0 else 'Dormant'
            stage_info = next((s for s in self.config['growth_stages'] if s['stage'] == growth_stage), 
                            self.config['growth_stages'][0])
            
            optimal = self.config['optimal_ranges']['nutrients']
            uptake_factor = stage_info['nutrient_uptake']
            
            # loop sensor entries
            zone_N_needs = []
            zone_P_needs = []
            zone_K_needs = []
            
            for _, sensor_row in zone_sensors.iterrows():
                sensor_id = sensor_row['sensor_id']
                sensor_N = sensor_row['nutrient_N']
                sensor_P = sensor_row['nutrient_P']
                sensor_K = sensor_row['nutrient_K']
                
                N_deficit = max(0, (optimal['N']['optimal'] - sensor_N) * uptake_factor * time_multiplier)
                P_deficit = max(0, (optimal['P']['optimal'] - sensor_P) * uptake_factor * time_multiplier)
                K_deficit = max(0, (optimal['K']['optimal'] - sensor_K) * uptake_factor * time_multiplier)
                
                sensor_fertilizer_needs[sensor_id] = {
                    'N': round(N_deficit, 2),
                    'P': round(P_deficit, 2),
                    'K': round(K_deficit, 2),
                    'zone_id': zone_id,
                    'current_N': round(sensor_N, 2),
                    'current_P': round(sensor_P, 2),
                    'current_K': round(sensor_K, 2)
                }
                
                zone_N_needs.append(N_deficit)
                zone_P_needs.append(P_deficit)
                zone_K_needs.append(K_deficit)
            
            # keep zone average
            fertilizer_needs[zone_id] = {
                'N': round(np.mean(zone_N_needs), 2) if zone_N_needs else 0,
                'P': round(np.mean(zone_P_needs), 2) if zone_P_needs else 0,
                'K': round(np.mean(zone_K_needs), 2) if zone_K_needs else 0
            }
        
        return fertilizer_needs, sensor_fertilizer_needs
    
    def calculate_water_only_cost(self, water_needs):
        """Calculate cost for water prescription only"""
        costs = self.config['costs']
        zone_area = 5000  # zone size about 5000 m2
        
        total_water = sum(water_needs.values()) * zone_area / 1000  # convert mm to m3
        water_cost = total_water * costs['water_per_m3']
        electricity_cost = total_water * costs['pumping_energy_per_m3'] * costs['electricity_per_kwh']
        
        return {
            'water_cost': round(water_cost, 2),
            'electricity_cost': round(electricity_cost, 2),
            'fertilizer_cost': 0.0,
            'total_cost': round(water_cost + electricity_cost, 2),
            'details': {
                'water_volume_m3': round(total_water, 2)
            }
        }
    
    def calculate_fertilizer_only_cost(self, fertilizer_needs):
        """Calculate cost for fertilizer prescription only"""
        costs = self.config['costs']
        zone_area = 5000  # zone size about 5000 m2
        
        total_N = sum(f['N'] for f in fertilizer_needs.values()) * zone_area / 1000
        total_P = sum(f['P'] for f in fertilizer_needs.values()) * zone_area / 1000
        total_K = sum(f['K'] for f in fertilizer_needs.values()) * zone_area / 1000
        
        N_cost = total_N * costs['fertilizer_N_per_kg']
        P_cost = total_P * costs['fertilizer_P_per_kg']
        K_cost = total_K * costs['fertilizer_K_per_kg']
        fertilizer_cost = N_cost + P_cost + K_cost
        
        return {
            'water_cost': 0.0,
            'electricity_cost': 0.0,
            'fertilizer_cost': round(fertilizer_cost, 2),
            'total_cost': round(fertilizer_cost, 2),
            'details': {
                'N_kg': round(total_N, 2),
                'N_cost': round(N_cost, 2),
                'P_kg': round(total_P, 2),
                'P_cost': round(P_cost, 2),
                'K_kg': round(total_K, 2),
                'K_cost': round(K_cost, 2)
            }
        }
    
    def calculate_cost(self, water_needs, fertilizer_needs):
        """Calculate total intervention cost"""
        costs = self.config['costs']
        
        # approximate zone area
        zone_area = 5000  # zone size about 5000 m2
        
        total_water = sum(water_needs.values()) * zone_area / 1000  # convert mm to m3
        water_cost = total_water * costs['water_per_m3']
        electricity_cost = total_water * costs['pumping_energy_per_m3'] * costs['electricity_per_kwh']
        
        total_N = sum(f['N'] for f in fertilizer_needs.values()) * zone_area / 1000
        total_P = sum(f['P'] for f in fertilizer_needs.values()) * zone_area / 1000
        total_K = sum(f['K'] for f in fertilizer_needs.values()) * zone_area / 1000
        
        N_cost = total_N * costs['fertilizer_N_per_kg']
        P_cost = total_P * costs['fertilizer_P_per_kg']
        K_cost = total_K * costs['fertilizer_K_per_kg']
        fertilizer_cost = N_cost + P_cost + K_cost
        
        return {
            'water_cost': round(water_cost, 2),
            'electricity_cost': round(electricity_cost, 2),
            'fertilizer_cost': round(fertilizer_cost, 2),
            'total_cost': round(water_cost + electricity_cost + fertilizer_cost, 2),
            'details': {
                'water_volume_m3': round(total_water, 2),
                'N_kg': round(total_N, 2),
                'N_cost': round(N_cost, 2),
                'P_kg': round(total_P, 2),
                'P_cost': round(P_cost, 2),
                'K_kg': round(total_K, 2),
                'K_cost': round(K_cost, 2)
            }
        }
    
    def generate_prescription(self, target_date_str):
        """Generate full prescription with heatmaps"""
        water_needs, sensor_water_needs = self.calculate_water_prescription(target_date_str)
        fertilizer_needs, sensor_fertilizer_needs = self.calculate_fertilizer_prescription(target_date_str)
        cost = self.calculate_cost(water_needs, fertilizer_needs)
        
        prescription = {
            'target_date': target_date_str,
            'water_prescription': water_needs,
            'fertilizer_prescription': fertilizer_needs,
            'sensor_water_needs': sensor_water_needs,
            'sensor_fertilizer_needs': sensor_fertilizer_needs,
            'cost_estimate': cost
        }
        
        with open('prescription.json', 'w') as f:
            json.dump(prescription, f, indent=2)
        
        return prescription

if __name__ == '__main__':
    dss = VineyardDSS()
    prescription = dss.generate_prescription('2024-12-08')
    print(json.dumps(prescription, indent=2))
