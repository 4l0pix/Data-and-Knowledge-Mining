# Vineyard DSS Project Documentation

This document describes the architecture, data flow, and implementation details of the Vineyard Decision Support System (DSS). The project simulates a precision agriculture platform that processes sensor measurements to provide irrigation and fertilization prescriptions for a multi-zone vineyard in Crete.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Configuration Files](#configuration-files)
   - [vineyard_config.json](#vineyard_configjson)
3. [Python Modules](#python-modules)
   - [data_generator.py](#data_generatorpy)
   - [dss_algorithm.py](#dss_algorithmpy)
   - [heatmap_generator.py](#heatmap_generatorpy)
   - [spatial_interpolation.py](#spatial_interpolationpy)
   - [app.py](#apppy)
4. [Frontend Application](#frontend-application)
   - [index.html Structure](#indexhtml-structure)
   - [JavaScript Logic](#javascript-logic)
   - [Styling Highlights](#styling-highlights)
5. [Data Files](#data-files)
6. [Typical Workflow](#typical-workflow)
7. [Extensibility Notes](#extensibility-notes)

---

## Project Overview

The DSS combines simulated climate, plant, and soil data to recommend interventions for five vineyard zones. The backend is written in Python (Flask) and exposes REST endpoints. The frontend is a single-page application built with HTML, CSS, Leaflet.js, and vanilla JavaScript.

Key capabilities:

- Generate realistic daily sensor measurements for 27 sensors over two years plus 30 forecast days.
- Model seasonal climates, zone microclimates, and sensor-level variations.
- Compute irrigation and fertilization requirements per sensor region.
- Render high-resolution contour heatmaps for measured data and prescriptions.
- Provide cost breakdowns for planned interventions.

---

## Configuration Files

### vineyard_config.json

Central configuration describing the vineyard layout and agronomic parameters.

**Sections:**

- `zones`: Geographic boundaries for Field1–Field5 with latitude/longitude polygon vertices. Each zone has a unique `zone_id` used throughout the codebase.
- `sensors`: Lists sensor metadata (ID, coordinates) grouped by zone. Sensors cover the vineyard with 50 m radius sampling areas.
- `optimal_ranges`: Target ranges for nitrogen (N), phosphorus (P), potassium (K), soil moisture, and pH.
- `costs`: Economic factors such as water price per cubic meter, electricity per kWh, and fertilizer prices per kilogram for each nutrient.
- `growth_stages`: Defines phenological stages (`Dormant`, `Bud Break`, `Flowering`, `Veraison`, `Harvest`). Each stage specifies a `water_factor` (relative water demand) and `nutrient_uptake` multiplier.

---

## Python Modules

### data_generator.py

Generates synthetic historical and forecast data.

**Key components:**

- `VineyardDataGenerator` class orchestrates data creation.
- Seasonal climate profiles (`seasonal_profiles`) define typical temperature and humidity ranges for Winter/Spring/Summer/Autumn.
- Zone-level microclimate offsets (`zone_microclimate`) introduce variation per field (±1.5 °C, ±8 % humidity).
- Sensor-level variations add additional random differences (±0.5 °C, ±3 % humidity) and soil characteristics (moisture retention, drainage, nutrient depletion rates).
- `_generate_weather`: Simulates daily temperature, humidity, rainfall, solar radiation by sampling within seasonal ranges and applying noise.
- `_generate_sensor_data`: For each sensor, projects soil moisture using a water-balance equation that accounts for rainfall, evapotranspiration, and drainage. Nutrients deplete according to sensor-specific rates and plant uptake multipliers.
- Outputs CSV files (`weather_data.csv`, `sensor_data.csv`, `plant_data.csv`, `intervention_data.csv`).

### dss_algorithm.py

Implements the decision logic.

**VineyardDSS class:**

- Loads configuration and sensor/plant/intervention datasets.
- `calculate_et(temp, solar, humidity)`: Computes evapotranspiration using a Hargreaves-based formulation adjusted for humidity.
- `calculate_water_prescription(target_date)`: 
  - Uses forecast weather leading up to the target date to estimate cumulative evapotranspiration and effective rainfall.
  - Retrieves predicted sensor readings on the target date.
  - Applies growth stage water factors, residual water from recent irrigation, and a progressive time multiplier (5 % per day ahead).
  - Returns zone-level averages plus `sensor_water_needs` mapping each sensor to required irrigation (mm).
- `calculate_fertilizer_prescription(target_date)`: 
  - Uses target date nutrient levels per sensor.
  - Applies growth stage uptake multipliers and time-based increase (6 % per day ahead).
  - Returns zone averages and `sensor_fertilizer_needs` (N, P, K deficits per sensor region).
- `calculate_water_only_cost`, `calculate_fertilizer_only_cost`, `calculate_cost`: Convert prescriptions into water/fertilizer volumes, then compute total expenditure using `costs` from configuration.
- `generate_prescription`: Combines water and fertilizer calculations, aggregates cost data, and writes `prescription.json` snapshot.

### heatmap_generator.py

Handles visualization of sensor measurements and prescriptions.

**HeatmapGenerator class:**

- `generate_contour_heatmap(date, data_type)`: 
  - Loads sensor measurements for a given metric (e.g., `ground_moisture`, `temperature`).
  - Combines sensor coordinates with zone boundary vertices to create interpolation points.
  - Uses SciPy `griddata` (linear + nearest) to interpolate values across a uniform grid.
  - Masks points outside vineyard zones and renders a contour plot with 30 levels using Matplotlib.
  - Returns PNG image encoded in base64 along with map bounds.
- `generate_sensor_prescription_heatmap(sensor_prescription_data, prescription_type)`: 
  - Builds interpolation inputs from per-sensor prescriptions (`water_mm` or `N`).
  - Adds zone boundary points with averaged values to smooth transitions.
  - Produces contour heatmaps highlighting areas requiring higher interventions.

### spatial_interpolation.py

Legacy module offering inverse distance weighting (IDW) interpolation. The main application now uses the Matplotlib-based approach from `heatmap_generator.py`, but this module remains for potential alternative interpolation strategies.

### app.py

Flask API server.

**Endpoints:**

- `GET /` serves the frontend (`index.html`).
- `GET /api/config` returns vineyard configuration.
- `POST /api/prescription` generates full water + fertilization prescriptions.
- `POST /api/prescription/water` and `/api/prescription/fertilizer` (if needed) handle single-purpose calculations.
- `GET /api/heatmap-image/<date>/<data_type>` produces sensor measurement heatmaps.
- `POST /api/prescription-heatmap` generates prescription heatmap overlays from sensor-level data.
- `GET /api/sensor-data/<date>` supplies raw sensor readings for UI popups.
- `POST /api/generate-data` (optional) re-runs the data generator.

The app ensures `sensor_data.csv` exists on startup by invoking `VineyardDataGenerator` when necessary.

---

## Frontend Application

### index.html Structure

A single-page interface with three primary sections:

1. **Sidebar Control Panel**
   - Data layer selector (radio buttons) for moisture, temperature, humidity, nutrients, pH.
   - DSS controls (target date input, full analysis trigger, reset button).
   - Prescription view toggle (water vs. fertilizer) shown after full analysis.
   - Cost breakdown display (total, water, electricity, fertilizer).
2. **Map Container** (`#map`)
   - Leaflet map centered on the vineyard with Esri satellite tiles.
3. **Modal/Popup Elements** (Leaflet popups attached to sensors/zones).

### JavaScript Logic

Key variables store map state (`map`, `zoneLayers`, `heatmapLayers`, `sensorMarkers`), data caches (`sensorData`, `sensorWaterNeeds`, `sensorFertilizerNeeds`), and UI state (`currentDate`, `currentDataType`, `currentPrescriptionView`).

Main functions:

- `init()`: Initializes map, fetches config, draws zones, sensors, and default heatmap.
- `drawZones()`: Adds Leaflet polygons for each field boundary with translucent styling.
- `drawSensors()`: Places sensor markers (red circle markers) and binds click handlers.
- `showSensorPopup(marker, sensorId)`: Displays current measurements and, when available, prescription recommendations for the specific sensor region.
- `handleLayerChange()`: Switches measurement heatmaps based on selected radio button.
- `showHeatmapLayer(dataType)`: Fetches and overlays measurement heatmap images.
- `calculateFullPrescription()`: Sends target date to `/api/prescription`, stores returned sensor-level prescriptions, updates cost display, reveals prescription view toggles, and calls `visualizePrescription()`.
- `handlePrescriptionViewChange()`: Switches between water and fertilizer heatmaps.
- `visualizePrescription()`: Generates prescription heatmap overlay via `/api/prescription-heatmap`, renders zone boundaries, and summarizes average needs in zone popups.
- `resetView()`: Clears map layers, hides prescription controls, resets cost display.

### Styling Highlights

The interface uses a glassmorphism theme with vineyard-inspired palette (`#F1F3E0`, `#D2DCB6`, `#A1BC98`, `#778873`, `#662222`). CSS applies backdrop blur, semi-transparent panels, and custom radio buttons (circular indicators without default browser styling).

---

## Data Files

Generated CSV datasets stored in the project root:

- `weather_data.csv`: Daily weather variables per zone.
- `sensor_data.csv`: Sensor-level measurements (temperature, humidity, moisture, nutrients, pH).
- `plant_data.csv`: Growth stage and phenology per zone.
- `intervention_data.csv`: Records of past irrigation/fertilization interventions (used to calculate residual effects).

These files are regenerated by `data_generator.py` if missing.

---

## Typical Workflow

1. **Initialize Data** (automatically if missing): `python data_generator.py`
2. **Run Backend Server**: `python app.py`
3. **Open Frontend**: Navigate to `http://127.0.0.1:5000/`
4. **Interact With UI**:
   - Choose a target visit date.
   - Click *Full Analysis* to compute prescriptions.
   - Toggle between *Water Needs* and *Fertilizer Needs* heatmaps.
   - Inspect individual sensors for localized recommendations.
   - Review cost estimates for planned interventions.

---

## Extensibility Notes

- **Additional Sensors/Zones**: Update `vineyard_config.json` and rerun the data generator.
- **Alternate Crops or Regions**: Adjust seasonal profiles, optimal ranges, and growth stages in the configuration file.
- **Different Cost Models**: Modify the `costs` section and related calculations in `dss_algorithm.py`.
- **New Visualization Layers**: Extend `heatmap_generator.py` and UI handlers to support additional metrics.
- **Real Data Integration**: Replace `data_generator.py` outputs with actual sensor feeds; ensure CSV schema remains consistent.

This documentation should serve as a comprehensive guide to the structure and behavior of the Vineyard DSS project.
