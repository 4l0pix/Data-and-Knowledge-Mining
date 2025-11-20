import json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

class SpatialInterpolator:
    def __init__(self, config_path='vineyard_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.epsilon = 1e-10
    
    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """Calculate distance in meters between two coordinates"""
        R = 6371000  # earth radius meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    def interpolate_value(self, lon, lat, sensor_data, data_col, baseline_value=None):
        """Interpolate value at (lon, lat) using inverse distance weighting"""
        weights = []
        values = []
        
        for zone_id, sensors in self.config['sensors'].items():
            for sensor in sensors:
                sensor_id = sensor['sensor_id']
                sensor_lon, sensor_lat = sensor['lon'], sensor['lat']
                radius = sensor['radius_m']
                
                dist = self.haversine_distance(lon, lat, sensor_lon, sensor_lat)
                
                if dist <= radius:
                    weight = 1.0 / (dist**2 + self.epsilon)
                    sensor_value = sensor_data[sensor_data['sensor_id'] == sensor_id][data_col].values
                    if len(sensor_value) > 0:
                        weights.append(weight)
                        values.append(sensor_value[0])
        
        if len(weights) > 0:
            total_weight = sum(weights)
            interpolated = sum(w * v for w, v in zip(weights, values)) / total_weight
            return interpolated
        elif baseline_value is not None:
            return baseline_value
        else:
            return 0
    
    def generate_heatmap(self, sensor_data, data_col, resolution=20, baseline=None):
        """Generate heatmap grid for visualization"""
        all_lons = [s['lon'] for sensors in self.config['sensors'].values() for s in sensors]
        all_lats = [s['lat'] for sensors in self.config['sensors'].values() for s in sensors]
        
        min_lon, max_lon = min(all_lons) - 0.001, max(all_lons) + 0.001
        min_lat, max_lat = min(all_lats) - 0.001, max(all_lats) + 0.001
        
        lons = np.linspace(min_lon, max_lon, resolution)
        lats = np.linspace(min_lat, max_lat, resolution)
        
        grid = []
        for lat in lats:
            row = []
            for lon in lons:
                value = self.interpolate_value(lon, lat, sensor_data, data_col, baseline)
                row.append(value)
            grid.append(row)
        
        return {
            'lons': lons.tolist(),
            'lats': lats.tolist(),
            'values': grid,
            'min': float(np.min(grid)),
            'max': float(np.max(grid))
        }
    
    def generate_all_heatmaps(self, date_str):
        """Generate all heatmap layers for a specific date"""
        sensor_data = pd.read_csv('sensor_data.csv')
        weather_data = pd.read_csv('weather_data.csv')
        plant_data = pd.read_csv('plant_data.csv')
        
        sensor_day = sensor_data[sensor_data['date'] == date_str]
        weather_day = weather_data[weather_data['date'] == date_str]
        plant_day = plant_data[plant_data['date'] == date_str]
        
        baseline_temp = weather_day['temperature'].values[0] if len(weather_day) > 0 else 20
        baseline_rain = weather_day['rainfall'].values[0] if len(weather_day) > 0 else 0
        
        heatmaps = {
            'date': date_str,
            'ground_moisture': self.generate_heatmap(sensor_day, 'ground_moisture'),
            'temperature': self.generate_heatmap(sensor_day, 'temperature', baseline_temp),
            'humidity': self.generate_heatmap(sensor_day, 'humidity', weather_day['humidity'].values[0] if len(weather_day) > 0 else 60),
            'pH': self.generate_heatmap(sensor_day, 'pH'),
            'nutrient_N': self.generate_heatmap(sensor_day, 'nutrient_N'),
            'nutrient_P': self.generate_heatmap(sensor_day, 'nutrient_P'),
            'nutrient_K': self.generate_heatmap(sensor_day, 'nutrient_K'),
            'rainfall': baseline_rain
        }
        
        # add zone health
        zone_health = {}
        for _, row in plant_day.iterrows():
            zone_health[row['zone_id']] = row['health_index']
        heatmaps['zone_health'] = zone_health
        
        return heatmaps

if __name__ == '__main__':
    interpolator = SpatialInterpolator()
    heatmaps = interpolator.generate_all_heatmaps('2024-12-01')
    with open('heatmaps_sample.json', 'w') as f:
        json.dump(heatmaps, f, indent=2)
    print("Generated sample heatmaps")
