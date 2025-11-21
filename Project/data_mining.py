import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # noninteractive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VineyardDataMiner:
    def __init__(self, config_path='vineyard_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
    def load_sensor_data(self, days_back=30):
        """load recent sensor data for mining"""
        sensor_data = pd.read_csv('sensor_data.csv')
        sensor_data['date'] = pd.to_datetime(sensor_data['date'])
        
        # get recent data
        cutoff_date = sensor_data['date'].max() - timedelta(days=days_back)
        recent_data = sensor_data[sensor_data['date'] >= cutoff_date].copy()
        
        return recent_data
    
    def detect_outliers_statistical(self, data, columns=None, z_threshold=3.0):
        """detect outliers using z-score and iqr methods"""
        if columns is None:
            columns = ['ground_moisture', 'temperature', 'humidity', 'nutrient_N', 'nutrient_P', 'nutrient_K', 'pH']
        
        outliers = pd.DataFrame()
        
        for col in columns:
            if col in data.columns:
                # z-score method
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                z_outliers = data[z_scores > z_threshold].copy()
                z_outliers['outlier_method'] = 'z_score'
                z_outliers['outlier_column'] = col
                z_outliers['outlier_value'] = z_scores[z_scores > z_threshold]
                
                # iqr method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].copy()
                iqr_outliers['outlier_method'] = 'iqr'
                iqr_outliers['outlier_column'] = col
                iqr_outliers['outlier_value'] = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                
                outliers = pd.concat([outliers, z_outliers, iqr_outliers], ignore_index=True)
        
        return outliers.drop_duplicates()
    
    def detect_outliers_isolation_forest(self, data, contamination=0.1):
        """detect outliers using isolation forest"""
        feature_cols = ['ground_moisture', 'temperature', 'humidity', 'nutrient_N', 'nutrient_P', 'nutrient_K', 'pH']
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if len(available_cols) < 3:
            return pd.DataFrame()
        
        # prepare features
        features = data[available_cols].dropna()
        if len(features) < 10:
            return pd.DataFrame()
        
        # fit isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        outlier_labels = iso_forest.fit_predict(features)
        
        # get outliers
        outlier_indices = features.index[outlier_labels == -1]
        outliers = data.loc[outlier_indices].copy()
        outliers['outlier_method'] = 'isolation_forest'
        outliers['outlier_score'] = iso_forest.score_samples(features)[outlier_labels == -1]
        
        return outliers
    
    def clean_outliers(self, data, method='cap', outlier_data=None):
        """clean outliers by removal, capping, or interpolation"""
        if outlier_data is None or len(outlier_data) == 0:
            return data
        
        cleaned_data = data.copy()
        
        if method == 'remove':
            # remove outlier rows
            outlier_indices = outlier_data.index.unique()
            cleaned_data = cleaned_data.drop(outlier_indices)
            
        elif method == 'cap':
            # cap values to percentiles
            for col in ['ground_moisture', 'temperature', 'humidity', 'nutrient_N', 'nutrient_P', 'nutrient_K', 'pH']:
                if col in cleaned_data.columns:
                    p1, p99 = cleaned_data[col].quantile([0.01, 0.99])
                    cleaned_data[col] = cleaned_data[col].clip(lower=p1, upper=p99)
                    
        elif method == 'interpolate':
            # interpolate outlier values
            for col in ['ground_moisture', 'temperature', 'humidity', 'nutrient_N', 'nutrient_P', 'nutrient_K', 'pH']:
                if col in cleaned_data.columns:
                    outlier_mask = cleaned_data.index.isin(outlier_data[outlier_data['outlier_column'] == col].index)
                    cleaned_data.loc[outlier_mask, col] = np.nan
                    cleaned_data[col] = cleaned_data.groupby('sensor_id')[col].interpolate(method='linear')
        
        return cleaned_data
    
    def cluster_sensors_spatial(self, data, n_clusters=None, method='kmeans'):
        """cluster sensors based on spatial patterns"""
        # aggregate sensor data
        sensor_features = data.groupby('sensor_id').agg({
            'ground_moisture': ['mean', 'std'],
            'temperature': ['mean', 'std'],
            'humidity': ['mean', 'std'],
            'nutrient_N': ['mean', 'std'],
            'nutrient_P': ['mean', 'std'],
            'nutrient_K': ['mean', 'std'],
            'pH': ['mean', 'std']
        }).fillna(0)
        
        # flatten column names
        sensor_features.columns = ['_'.join(col).strip() for col in sensor_features.columns]
        
        if len(sensor_features) < 3:
            return None
        
        # scale features
        features_scaled = self.scaler.fit_transform(sensor_features)
        
        if method == 'kmeans':
            if n_clusters is None:
                # find optimal clusters using elbow method
                n_clusters = self._find_optimal_clusters(features_scaled, max_k=min(8, len(sensor_features)-1))
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
        # add sensor coordinates
        sensor_coords = {}
        for zone_id, sensors in self.config['sensors'].items():
            for sensor in sensors:
                sensor_coords[sensor['sensor_id']] = {
                    'lon': sensor['lon'],
                    'lat': sensor['lat'],
                    'zone_id': zone_id
                }
        
        # create results
        cluster_results = []
        for i, (sensor_id, cluster_id) in enumerate(zip(sensor_features.index, cluster_labels)):
            coords = sensor_coords.get(sensor_id, {})
            cluster_results.append({
                'sensor_id': sensor_id,
                'cluster_id': int(cluster_id),
                'zone_id': coords.get('zone_id', 'unknown'),
                'lon': coords.get('lon', 0),
                'lat': coords.get('lat', 0),
                **dict(zip(sensor_features.columns, sensor_features.loc[sensor_id]))
            })
        
        return pd.DataFrame(cluster_results)
    
    def cluster_temporal_patterns(self, data, n_clusters=4):
        """cluster based on temporal patterns"""
        # create daily aggregates per sensor
        daily_patterns = data.groupby(['sensor_id', data['date'].dt.date]).agg({
            'ground_moisture': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'nutrient_N': 'mean'
        }).reset_index()
        
        # pivot to create time series features
        pivot_data = daily_patterns.pivot(index='sensor_id', columns='date', values='ground_moisture')
        pivot_data = pivot_data.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).fillna(0)
        
        if len(pivot_data) < 3 or pivot_data.shape[1] < 3:
            return None
        
        # apply pca for dimensionality reduction
        pca = PCA(n_components=min(5, pivot_data.shape[1]))
        pca_features = pca.fit_transform(pivot_data)
        
        # cluster temporal patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_features)
        
        results = []
        for sensor_id, cluster_id in zip(pivot_data.index, cluster_labels):
            results.append({
                'sensor_id': sensor_id,
                'temporal_cluster': int(cluster_id),
                'pattern_variance': np.var(pivot_data.loc[sensor_id].values),
                'pattern_trend': np.polyfit(range(len(pivot_data.columns)), pivot_data.loc[sensor_id].values, 1)[0]
            })
        
        return pd.DataFrame(results)
    
    def find_anomalous_zones(self, data, threshold_percentile=95):
        """identify zones with unusual patterns"""
        zone_stats = data.groupby('zone_id').agg({
            'ground_moisture': ['mean', 'std', 'min', 'max'],
            'temperature': ['mean', 'std', 'min', 'max'],
            'humidity': ['mean', 'std', 'min', 'max'],
            'nutrient_N': ['mean', 'std', 'min', 'max']
        })
        
        # flatten columns
        zone_stats.columns = ['_'.join(col).strip() for col in zone_stats.columns]
        
        # calculate anomaly scores
        anomaly_scores = {}
        for zone_id in zone_stats.index:
            # composite score based on deviation from median
            score = 0
            for col in zone_stats.columns:
                if 'std' in col:  # high variability indicator
                    zone_val = zone_stats.loc[zone_id, col]
                    median_val = zone_stats[col].median()
                    score += abs(zone_val - median_val) / (median_val + 1e-8)
            
            anomaly_scores[zone_id] = score
        
        # identify anomalous zones
        threshold = np.percentile(list(anomaly_scores.values()), threshold_percentile)
        anomalous_zones = [zone for zone, score in anomaly_scores.items() if score > threshold]
        
        return {
            'anomalous_zones': anomalous_zones,
            'anomaly_scores': anomaly_scores,
            'threshold': threshold
        }
    
    def detect_sensor_drift(self, data, window_days=7):
        """detect sensors showing systematic drift"""
        drift_results = []
        
        for sensor_id in data['sensor_id'].unique():
            sensor_data = data[data['sensor_id'] == sensor_id].sort_values('date')
            
            if len(sensor_data) < window_days:
                continue
                
            # check for drift in key metrics
            for metric in ['ground_moisture', 'temperature', 'nutrient_N', 'pH']:
                if metric in sensor_data.columns:
                    values = sensor_data[metric].values
                    if len(values) > 5:
                        # linear trend test
                        x = np.arange(len(values))
                        slope, _, r_value, p_value, _ = stats.linregress(x, values)
                        
                        # significant trend indicates drift
                        if p_value < 0.05 and abs(r_value) > 0.7:
                            drift_results.append({
                                'sensor_id': sensor_id,
                                'metric': metric,
                                'drift_slope': slope,
                                'correlation': r_value,
                                'p_value': p_value,
                                'severity': 'high' if abs(slope) > 1.0 else 'moderate'
                            })
        
        return pd.DataFrame(drift_results)
    
    def generate_mining_report(self, data, output_path='mining_report.json'):
        """generate comprehensive data mining report"""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': data['date'].min().isoformat(),
                'end_date': data['date'].max().isoformat(),
                'total_records': len(data),
                'sensors_count': data['sensor_id'].nunique()
            }
        }
        
        # outlier detection
        print("detecting outliers...")
        statistical_outliers = self.detect_outliers_statistical(data)
        isolation_outliers = self.detect_outliers_isolation_forest(data)
        
        report['outliers'] = {
            'statistical_outliers_count': len(statistical_outliers),
            'isolation_outliers_count': len(isolation_outliers),
            'total_affected_sensors': pd.concat([statistical_outliers, isolation_outliers])['sensor_id'].nunique() if len(statistical_outliers) > 0 or len(isolation_outliers) > 0 else 0
        }
        
        # clustering
        print("performing spatial clustering...")
        spatial_clusters = self.cluster_sensors_spatial(data)
        temporal_clusters = self.cluster_temporal_patterns(data)
        
        if spatial_clusters is not None:
            report['spatial_clustering'] = {
                'clusters_found': spatial_clusters['cluster_id'].nunique(),
                'cluster_distribution': spatial_clusters['cluster_id'].value_counts().to_dict()
            }
        
        if temporal_clusters is not None:
            report['temporal_clustering'] = {
                'pattern_clusters': temporal_clusters['temporal_cluster'].nunique(),
                'high_variance_sensors': len(temporal_clusters[temporal_clusters['pattern_variance'] > temporal_clusters['pattern_variance'].quantile(0.8)])
            }
        
        # anomaly detection
        print("finding anomalous zones...")
        zone_anomalies = self.find_anomalous_zones(data)
        sensor_drift = self.detect_sensor_drift(data)
        
        report['anomalies'] = {
            'anomalous_zones': zone_anomalies['anomalous_zones'],
            'zone_anomaly_scores': zone_anomalies['anomaly_scores'],
            'sensors_with_drift': len(sensor_drift),
            'high_severity_drift': len(sensor_drift[sensor_drift['severity'] == 'high']) if len(sensor_drift) > 0 else 0
        }
        
        # data quality metrics
        report['data_quality'] = {}
        for col in ['ground_moisture', 'temperature', 'humidity', 'nutrient_N', 'nutrient_P', 'nutrient_K', 'pH']:
            if col in data.columns:
                report['data_quality'][col] = {
                    'missing_percentage': (data[col].isna().sum() / len(data)) * 100,
                    'outlier_percentage': (len(statistical_outliers[statistical_outliers['outlier_column'] == col]) / len(data)) * 100,
                    'coefficient_variation': (data[col].std() / data[col].mean()) * 100 if data[col].mean() != 0 else 0
                }
        
        # recommendations
        recommendations = []
        if len(statistical_outliers) > len(data) * 0.05:
            recommendations.append("high outlier rate detected - consider sensor calibration")
        if len(zone_anomalies['anomalous_zones']) > 0:
            recommendations.append(f"investigate anomalous zones: {zone_anomalies['anomalous_zones']}")
        if len(sensor_drift) > 0:
            recommendations.append("sensor drift detected - schedule maintenance")
        
        report['recommendations'] = recommendations
        
        # save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _find_optimal_clusters(self, features, max_k=8):
        """find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:
                sil_score = silhouette_score(features, kmeans.labels_)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # find elbow point
        if len(silhouette_scores) > 0:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3  # default
        
        return optimal_k

if __name__ == '__main__':
    miner = VineyardDataMiner()
    data = miner.load_sensor_data(days_back=30)
    report = miner.generate_mining_report(data)
    print("data mining report generated")
    print(f"found {report['outliers']['statistical_outliers_count']} statistical outliers")
    print(f"identified {len(report['anomalies']['anomalous_zones'])} anomalous zones")