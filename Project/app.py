from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
from dss_algorithm import VineyardDSS
from spatial_interpolation import SpatialInterpolator
from data_generator import VineyardDataGenerator
from heatmap_generator import HeatmapGenerator
from data_mining import VineyardDataMiner
import os

app = Flask(__name__, static_folder='.')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/config')
def get_config():
    with open('vineyard_config.json', 'r') as f:
        return jsonify(json.load(f))

@app.route('/api/heatmap/<date>')
def get_heatmap(date):
    interpolator = SpatialInterpolator()
    heatmaps = interpolator.generate_all_heatmaps(date)
    return jsonify(heatmaps)

@app.route('/api/prescription', methods=['POST'])
def calculate_prescription():
    data = request.json
    target_date = data.get('target_date')
    
    dss = VineyardDSS()
    prescription = dss.generate_prescription(target_date)
    return jsonify(prescription)

@app.route('/api/prescription/water', methods=['POST'])
def calculate_water_prescription():
    data = request.json
    target_date = data.get('target_date')
    
    dss = VineyardDSS()
    water_needs, sensor_water_needs = dss.calculate_water_prescription(target_date)
    cost = dss.calculate_water_only_cost(water_needs)
    
    return jsonify({
        'target_date': target_date,
        'water_prescription': water_needs,
        'sensor_water_needs': sensor_water_needs,
        'cost_estimate': cost
    })

@app.route('/api/prescription/fertilizer', methods=['POST'])
def calculate_fertilizer_prescription():
    data = request.json
    target_date = data.get('target_date')
    
    dss = VineyardDSS()
    fertilizer_needs, sensor_fertilizer_needs = dss.calculate_fertilizer_prescription(target_date)
    cost = dss.calculate_fertilizer_only_cost(fertilizer_needs)
    
    return jsonify({
        'target_date': target_date,
        'fertilizer_prescription': fertilizer_needs,
        'sensor_fertilizer_needs': sensor_fertilizer_needs,
        'cost_estimate': cost
    })

@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    gen = VineyardDataGenerator()
    gen.generate_all_data()
    return jsonify({'status': 'success', 'message': 'Data generated'})

@app.route('/api/heatmap-image/<date>/<data_type>')
def get_heatmap_image(date, data_type):
    gen = HeatmapGenerator()
    heatmap = gen.generate_contour_heatmap(date, data_type)
    if heatmap:
        return jsonify(heatmap)
    return jsonify({'error': 'Failed to generate heatmap'}), 404

@app.route('/api/prescription-heatmap', methods=['POST'])
def get_prescription_heatmap():
    data = request.json
    sensor_prescription_data = data.get('sensor_prescription_data')
    prescription_type = data.get('prescription_type', 'water')
    
    gen = HeatmapGenerator()
    heatmap = gen.generate_sensor_prescription_heatmap(sensor_prescription_data, prescription_type)
    return jsonify(heatmap)

@app.route('/api/sensor-data/<date>')
def get_sensor_data(date):
    import pandas as pd
    try:
        sensor_data = pd.read_csv('sensor_data.csv')
        date_data = sensor_data[sensor_data['date'] == date]
        return jsonify(date_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/outliers', methods=['POST'])
def detect_outliers():
    data = request.json
    days_back = data.get('days_back', 30)
    method = data.get('method', 'statistical')  # 'statistical' or 'isolation_forest'
    
    miner = VineyardDataMiner()
    sensor_data = miner.load_sensor_data(days_back=days_back)
    
    if method == 'statistical':
        outliers = miner.detect_outliers_statistical(sensor_data)
    else:
        outliers = miner.detect_outliers_isolation_forest(sensor_data)
    
    return jsonify({
        'outliers_count': len(outliers),
        'outliers': outliers.to_dict('records') if len(outliers) > 0 else [],
        'affected_sensors': outliers['sensor_id'].nunique() if len(outliers) > 0 else 0
    })

@app.route('/api/mining/clusters', methods=['POST'])
def cluster_sensors():
    data = request.json
    days_back = data.get('days_back', 30)
    cluster_type = data.get('type', 'spatial')  # 'spatial' or 'temporal'
    n_clusters = data.get('n_clusters', None)
    
    miner = VineyardDataMiner()
    sensor_data = miner.load_sensor_data(days_back=days_back)
    
    if cluster_type == 'spatial':
        clusters = miner.cluster_sensors_spatial(sensor_data, n_clusters=n_clusters)
    else:
        clusters = miner.cluster_temporal_patterns(sensor_data, n_clusters=n_clusters or 4)
    
    if clusters is not None:
        return jsonify({
            'clusters': clusters.to_dict('records'),
            'cluster_count': clusters.iloc[:, 1].nunique(),  # second column is cluster id
            'success': True
        })
    else:
        return jsonify({'success': False, 'message': 'insufficient data for clustering'})

@app.route('/api/mining/anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json
    days_back = data.get('days_back', 30)
    
    miner = VineyardDataMiner()
    sensor_data = miner.load_sensor_data(days_back=days_back)
    
    zone_anomalies = miner.find_anomalous_zones(sensor_data)
    sensor_drift = miner.detect_sensor_drift(sensor_data)
    
    return jsonify({
        'anomalous_zones': zone_anomalies['anomalous_zones'],
        'zone_scores': zone_anomalies['anomaly_scores'],
        'sensor_drift': sensor_drift.to_dict('records') if len(sensor_drift) > 0 else [],
        'drift_count': len(sensor_drift)
    })

@app.route('/api/mining/clean-data', methods=['POST'])
def clean_data():
    data = request.json
    days_back = data.get('days_back', 30)
    method = data.get('method', 'cap')  # 'remove', 'cap', 'interpolate'
    
    miner = VineyardDataMiner()
    sensor_data = miner.load_sensor_data(days_back=days_back)
    
    # detect outliers first
    outliers = miner.detect_outliers_statistical(sensor_data)
    
    # clean data
    cleaned_data = miner.clean_outliers(sensor_data, method=method, outlier_data=outliers)
    
    return jsonify({
        'original_records': len(sensor_data),
        'cleaned_records': len(cleaned_data),
        'outliers_processed': len(outliers),
        'cleaning_method': method,
        'success': True
    })

@app.route('/api/mining/report', methods=['POST'])
def generate_mining_report():
    data = request.json
    days_back = data.get('days_back', 30)
    
    miner = VineyardDataMiner()
    sensor_data = miner.load_sensor_data(days_back=days_back)
    
    try:
        report = miner.generate_mining_report(sensor_data)
        return jsonify({
            'success': True,
            'report': report,
            'file_saved': 'mining_report.json'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    if not os.path.exists('sensor_data.csv'):
        print("Generating initial data...")
        gen = VineyardDataGenerator()
        gen.generate_all_data()
    app.run(debug=True, port=5000)
