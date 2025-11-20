import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # noninteractive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns
import io
import base64
from spatial_interpolation import SpatialInterpolator
from scipy.interpolate import griddata

class HeatmapGenerator:
    def __init__(self, config_path='vineyard_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.interpolator = SpatialInterpolator(config_path)
    
    def generate_contour_heatmap(self, date_str, data_type='ground_moisture', resolution=200):
        """Generate high-quality contour heatmap clipped to zone boundaries as base64 image"""
        sensor_data = pd.read_csv('sensor_data.csv')
        sensor_day = sensor_data[sensor_data['date'] == date_str]
        
        if len(sensor_day) == 0:
            return None
        
        # collect sensor points
        sensor_points = []
        
        for zone_id, sensors in self.config['sensors'].items():
            for sensor in sensors:
                sensor_id = sensor['sensor_id']
                sensor_reading = sensor_day[sensor_day['sensor_id'] == sensor_id]
                if len(sensor_reading) > 0 and data_type in sensor_reading.columns:
                    value = sensor_reading[data_type].values[0]
                    sensor_points.append({
                        'lon': sensor['lon'],
                        'lat': sensor['lat'],
                        'value': value,
                        'zone_id': zone_id
                    })
        
        if len(sensor_points) == 0:
            return None
        
        # build interpolation arrays
        lons = np.array([p['lon'] for p in sensor_points])
        lats = np.array([p['lat'] for p in sensor_points])
        values = np.array([p['value'] for p in sensor_points])
        
        # add boundary points for coverage
        boundary_lons = []
        boundary_lats = []
        boundary_values = []
        
        for zone_name, zone_config in self.config['zones'].items():
            boundary = zone_config['boundary']
            zone_id = zone_config['zone_id']
            
            # average sensor values per zone
            zone_sensor_values = [p['value'] for p in sensor_points if p['zone_id'] == zone_id]
            if len(zone_sensor_values) > 0:
                avg_zone_value = np.mean(zone_sensor_values)
                
                # add boundary vertices
                for vertex in boundary:
                    boundary_lons.append(vertex['lon'])
                    boundary_lats.append(vertex['lat'])
                    boundary_values.append(avg_zone_value)
        
        # merge sensors and boundaries
        all_lons = np.concatenate([lons, boundary_lons])
        all_lats = np.concatenate([lats, boundary_lats])
        all_values = np.concatenate([values, boundary_values])
        
        # pad bounds for coverage
        lon_min, lon_max = np.min(all_lons) - 0.0005, np.max(all_lons) + 0.0005
        lat_min, lat_max = np.min(all_lats) - 0.0005, np.max(all_lats) + 0.0005
        
        # build grid
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        
        # interpolate grid values
        points = np.column_stack((all_lons, all_lats))
        
        # prefer linear interpolation
        grid_values = griddata(points, all_values, (grid_lon_2d, grid_lat_2d), method='linear')
        
        # fill nan with nearest
        nan_mask = np.isnan(grid_values)
        if np.any(nan_mask):
            grid_values_nearest = griddata(points, all_values, (grid_lon_2d, grid_lat_2d), method='nearest')
            grid_values = np.where(nan_mask, grid_values_nearest, grid_values)
        
        # mask outside zones
        mask = np.ones_like(grid_values, dtype=bool)
        for i in range(resolution):
            for j in range(resolution):
                point_lon = grid_lon_2d[i, j]
                point_lat = grid_lat_2d[i, j]
                in_any_zone = False
                
                for zone_name, zone_config in self.config['zones'].items():
                    boundary = zone_config['boundary']
                    vertices = [(p['lon'], p['lat']) for p in boundary]
                    path = Path(vertices)
                    if path.contains_point((point_lon, point_lat)):
                        in_any_zone = True
                        break
                
                if not in_any_zone:
                    mask[i, j] = False
        
        grid_values_masked = np.ma.masked_where(mask == False, grid_values)
        
        # build figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        # custom color ramp
        colors = ['#00ff00', '#88ff00', '#ffff00', '#ff8800', '#ff0000']
        cmap = LinearSegmentedColormap.from_list('vineyard', colors, N=256)
        
        # draw filled contours
        vmin = np.nanmin(grid_values_masked)
        vmax = np.nanmax(grid_values_masked)
        
        if vmax - vmin < 0.01:  # widen flat range
            vmin -= 0.5
            vmax += 0.5
        
        levels = np.linspace(vmin, vmax, 30)
        contourf = ax.contourf(grid_lon_2d, grid_lat_2d, grid_values_masked, 
                               levels=levels, cmap=cmap, alpha=0.7, antialiased=True)
        
        # add contour lines
        contour_lines = ax.contour(grid_lon_2d, grid_lat_2d, grid_values_masked, 
                                   levels=np.linspace(vmin, vmax, 15), 
                                   colors='black', alpha=0.15, linewidths=0.5, antialiased=True)
        
        # draw zone borders
        for zone_name, zone_config in self.config['zones'].items():
            boundary = zone_config['boundary']
            boundary_lons = [p['lon'] for p in boundary] + [boundary[0]['lon']]
            boundary_lats = [p['lat'] for p in boundary] + [boundary[0]['lat']]
            ax.plot(boundary_lons, boundary_lats, 'white', linewidth=2.5, alpha=0.9, zorder=4)
        
        # draw sensor markers
        ax.scatter(lons, lats, c='red', s=15, marker='o', 
                  edgecolors='white', linewidths=0.8, zorder=5)
        
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0, transparent=True, dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            'image': img_base64,
            'bounds': [[lat_min, lon_min], [lat_max, lon_max]],
            'min_value': float(np.nanmin(values)),
            'max_value': float(np.nanmax(values)),
            'data_type': data_type
        }
    
    def generate_prescription_heatmap(self, prescription_data, prescription_type='water', resolution=200):
        """Generate high-quality contour heatmap for water or fertilizer prescriptions clipped to zones"""
        # build zone interpolation set
        interp_lons = []
        interp_lats = []
        interp_values = []
        values = []
        
        for zone_name, zone_config in self.config['zones'].items():
            zone_id = zone_config['zone_id']
            
            if prescription_type == 'water':
                value = prescription_data.get(zone_id, 0)
            else:  # fertilizer
                fert_data = prescription_data.get(zone_id, {})
                value = fert_data.get('N', 0)
            
            if value > 0:
                values.append(value)
                boundary = zone_config['boundary']
                zone_lons = [p['lon'] for p in boundary]
                zone_lats = [p['lat'] for p in boundary]
                center_lon = sum(zone_lons) / len(zone_lons)
                center_lat = sum(zone_lats) / len(zone_lats)
                
                # add center point and edges
                interp_lons.append(center_lon)
                interp_lats.append(center_lat)
                interp_values.append(value)
                
                # add boundary points
                for p in boundary:
                    interp_lons.append(p['lon'])
                    interp_lats.append(p['lat'])
                    interp_values.append(value)
        
        if len(values) == 0:
            return None
        
        # build bounds
        all_lons = [p['lon'] for zone_config in self.config['zones'].values() for p in zone_config['boundary']]
        all_lats = [p['lat'] for zone_config in self.config['zones'].values() for p in zone_config['boundary']]
        lon_min, lon_max = min(all_lons) - 0.001, max(all_lons) + 0.001
        lat_min, lat_max = min(all_lats) - 0.001, max(all_lats) + 0.001
        
        # create grid
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        
        # interpolate grid
        points = np.column_stack((interp_lons, interp_lats))
        grid_values = griddata(points, interp_values, (grid_lon_2d, grid_lat_2d), method='cubic')
        
        # mask outside zones
        mask = np.ones_like(grid_values, dtype=bool)
        for i in range(resolution):
            for j in range(resolution):
                point_lon = grid_lon_2d[i, j]
                point_lat = grid_lat_2d[i, j]
                in_any_zone = False
                
                for zone_name, zone_config in self.config['zones'].items():
                    boundary = zone_config['boundary']
                    vertices = [(p['lon'], p['lat']) for p in boundary]
                    path = Path(vertices)
                    if path.contains_point((point_lon, point_lat)):
                        in_any_zone = True
                        break
                
                if not in_any_zone:
                    mask[i, j] = False
        
        grid_values_masked = np.ma.masked_where(mask == False, grid_values)
        
        # build figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        # custom color ramp
        colors = ['#00ff00', '#88ff00', '#ffff00', '#ff8800', '#ff0000']
        cmap = LinearSegmentedColormap.from_list('prescription', colors, N=256)
        
        # draw filled contours
        vmin = np.nanmin(grid_values_masked)
        vmax = np.nanmax(grid_values_masked)
        
        if vmax - vmin < 0.01:
            vmin -= 0.5
            vmax += 0.5
        
        levels = np.linspace(vmin, vmax, 25)
        contourf = ax.contourf(grid_lon_2d, grid_lat_2d, grid_values_masked, 
                               levels=levels, cmap=cmap, alpha=0.75, antialiased=True)
        
        # draw zone borders
        for zone_name, zone_config in self.config['zones'].items():
            boundary = zone_config['boundary']
            boundary_lons = [p['lon'] for p in boundary] + [boundary[0]['lon']]
            boundary_lats = [p['lat'] for p in boundary] + [boundary[0]['lat']]
            ax.plot(boundary_lons, boundary_lats, 'white', linewidth=2.5, alpha=0.9, zorder=4)
        
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0, transparent=True, dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            'image': img_base64,
            'bounds': [[lat_min, lon_min], [lat_max, lon_max]],
            'min_value': float(min(values)),
            'max_value': float(max(values)),
            'prescription_type': prescription_type
        }

    def generate_sensor_prescription_heatmap(self, sensor_prescription_data, prescription_type='water', resolution=200):
        """Generate high-quality contour heatmap using sensor-level prescription data"""
        if not sensor_prescription_data:
            return None
        
        # gather sensor coordinates
        interp_lons = []
        interp_lats = []
        interp_values = []
        
        for sensor_id, sensor_needs in sensor_prescription_data.items():
            # find sensor location
            sensor_config = None
            for zone_config in self.config['sensors'].values():
                for sensor in zone_config:
                    if sensor['sensor_id'] == sensor_id:
                        sensor_config = sensor
                        break
                if sensor_config:
                    break
            
            if sensor_config:
                if prescription_type == 'water':
                    value = sensor_needs.get('water_mm', 0)
                else:  # fertilizer uses n value
                    value = sensor_needs.get('N', 0)
                
                interp_lons.append(sensor_config['lon'])
                interp_lats.append(sensor_config['lat'])
                interp_values.append(value)
        
        if len(interp_values) == 0:
            return None
        
        # add zone boundaries with averages
        for zone_name, zone_config in self.config['zones'].items():
            zone_id = zone_config['zone_id']
            # compute zone average
            zone_sensor_values = []
            for sensor_id, sensor_needs in sensor_prescription_data.items():
                if sensor_needs.get('zone_id') == zone_id:
                    if prescription_type == 'water':
                        zone_sensor_values.append(sensor_needs.get('water_mm', 0))
                    else:
                        zone_sensor_values.append(sensor_needs.get('N', 0))
            
            if zone_sensor_values:
                avg_value = sum(zone_sensor_values) / len(zone_sensor_values)
                for point in zone_config['boundary']:
                    interp_lons.append(point['lon'])
                    interp_lats.append(point['lat'])
                    interp_values.append(avg_value)

        # build bounds
        all_lons = [p['lon'] for zone_config in self.config['zones'].values() for p in zone_config['boundary']]
        all_lats = [p['lat'] for zone_config in self.config['zones'].values() for p in zone_config['boundary']]
        lon_min, lon_max = min(all_lons) - 0.001, max(all_lons) + 0.001
        lat_min, lat_max = min(all_lats) - 0.001, max(all_lats) + 0.001
        
        # create grid
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        
        # interpolate values
        points = np.column_stack((interp_lons, interp_lats))
        grid_values = griddata(points, interp_values, (grid_lon_2d, grid_lat_2d), method='linear')
        
        # fill nan with nearest
        nan_mask = np.isnan(grid_values)
        if np.any(nan_mask):
            grid_values_nearest = griddata(points, interp_values, (grid_lon_2d, grid_lat_2d), method='nearest')
            grid_values[nan_mask] = grid_values_nearest[nan_mask]
        
        # mask outside zones
        mask = np.ones_like(grid_values, dtype=bool)
        for i in range(resolution):
            for j in range(resolution):
                point_lon = grid_lon_2d[i, j]
                point_lat = grid_lat_2d[i, j]
                in_any_zone = False
                
                for zone_name, zone_config in self.config['zones'].items():
                    boundary = zone_config['boundary']
                    vertices = [(p['lon'], p['lat']) for p in boundary]
                    path = Path(vertices)
                    if path.contains_point((point_lon, point_lat)):
                        in_any_zone = True
                        break
                
                if not in_any_zone:
                    mask[i, j] = False
        
        grid_values_masked = np.ma.masked_where(mask == False, grid_values)
        
        # build figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        # custom color ramp
        colors = ['#00ff00', '#88ff00', '#ffff00', '#ff8800', '#ff0000']
        cmap = LinearSegmentedColormap.from_list('prescription', colors, N=256)
        
        # draw filled contours
        vmin = np.nanmin(grid_values_masked)
        vmax = np.nanmax(grid_values_masked)
        
        if vmax - vmin < 0.01:
            vmin -= 0.5
            vmax += 0.5
        
        levels = np.linspace(vmin, vmax, 30)
        contourf = ax.contourf(grid_lon_2d, grid_lat_2d, grid_values_masked, 
                               levels=levels, cmap=cmap, alpha=0.75, antialiased=True)
        
        # add contour lines
        contour = ax.contour(grid_lon_2d, grid_lat_2d, grid_values_masked,
                            levels=15, colors='white', linewidths=0.5, alpha=0.3)
        
        # draw sensor markers
        sensor_lons = []
        sensor_lats = []
        for zone_config in self.config['sensors'].values():
            for sensor in zone_config:
                sensor_lons.append(sensor['lon'])
                sensor_lats.append(sensor['lat'])
        
        ax.scatter(sensor_lons, sensor_lats, c='red', s=15, alpha=0.8, zorder=5, edgecolors='white', linewidths=0.5)
        
        # draw zone borders
        for zone_name, zone_config in self.config['zones'].items():
            boundary = zone_config['boundary']
            boundary_lons = [p['lon'] for p in boundary] + [boundary[0]['lon']]
            boundary_lats = [p['lat'] for p in boundary] + [boundary[0]['lat']]
            ax.plot(boundary_lons, boundary_lats, 'white', linewidth=2.5, alpha=0.9, zorder=4)
        
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0, transparent=True, dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            'image': img_base64,
            'bounds': [[lat_min, lon_min], [lat_max, lon_max]],
            'min_value': float(min(interp_values)),
            'max_value': float(max(interp_values)),
            'prescription_type': prescription_type
        }

if __name__ == '__main__':
    gen = HeatmapGenerator()
    heatmap = gen.generate_contour_heatmap('2024-12-01', 'ground_moisture')
    if heatmap:
        print(f"Generated heatmap with bounds: {heatmap['bounds']}")
        print(f"Value range: {heatmap['min_value']:.2f} - {heatmap['max_value']:.2f}")
