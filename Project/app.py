from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
from dss_algorithm import VineyardDSS
from spatial_interpolation import SpatialInterpolator
from data_generator import VineyardDataGenerator
from heatmap_generator import HeatmapGenerator
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

if __name__ == '__main__':
    if not os.path.exists('sensor_data.csv'):
        print("Generating initial data...")
        gen = VineyardDataGenerator()
        gen.generate_all_data()
    app.run(debug=True, port=5000)
