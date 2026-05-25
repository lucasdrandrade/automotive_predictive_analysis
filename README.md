# Engine Health Analytics Dashboard

A local-first web dashboard for automotive predictive maintenance. It reads engine telemetry from a CSV file and provides interactive visualizations to explore sensor data and detect fault patterns.

---

## How It Works

The app is built with **Plotly Dash** and follows a three-module structure:

| File | Responsibility |
|---|---|
| `data_loader.py` | Reads and normalizes `engine_data.csv` into a Pandas DataFrame; computes summary statistics |
| `charts.py` | Builds all Plotly figures (donut, bar, box plots, heatmap, scatter, line chart) |
| `app.py` | Defines the Dash layout, KPI cards, filter controls, and reactive callbacks |

### Data

The dataset (`Data_Set/engine_data.csv`) contains engine telemetry readings with the following columns:

| Column | Description |
|---|---|
| Engine rpm | Crankshaft revolutions per minute |
| Lub oil pressure | Lubrication oil pressure |
| Fuel pressure | Fuel rail pressure |
| Coolant pressure | Cooling system pressure |
| lub oil temp | Lubrication oil temperature (°C) |
| Coolant temp | Coolant temperature (°C) |
| Engine Condition | `1` = Healthy, `0` = Faulty |

### Dashboard Sections

- **KPI Cards** — total readings, healthy count, faulty count, and fault rate for the current filter selection
- **Condition Donut** — proportion of healthy vs. faulty readings
- **Normalized Sensor Means** — grouped bar chart comparing average sensor values between conditions (normalized 0–1)
- **Sensor Box Plots** — distribution spread for each sensor, split by condition
- **Correlation Heatmap** — Pearson correlation matrix across all sensors and engine condition
- **Scatter Plot** — any two sensors plotted against each other; X/Y axes are user-selectable
- **Sequential Line Chart** — sensor values over the reading index; sensor is user-selectable
- **Data Explorer** — paginated, sortable, filterable raw data table (capped at 500 rows)

### Filters

All charts and KPIs react live to two global filters:

- **Engine Condition** — show All, Healthy only, or Faulty only readings
- **RPM Range** — a range slider to narrow the dataset to a specific RPM band

---

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:

```
dash>=2.14.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
dash-bootstrap-components>=1.5.0
```

---

## Running the App

From inside the `Automtive_Prediction_Tool` directory:

```bash
python app.py
```

Then open your browser at:

```
http://localhost:8050
```

To stop the app, press `Ctrl+C` in the terminal.

---

## Project Structure

```
Automtive_Prediction_Tool/
├── app.py              # Dash app entry point — layout and callbacks
├── charts.py           # Plotly figure builders
├── data_loader.py      # CSV ingestion and statistics
├── requirements.txt    # Python dependencies
└── Data_Set/
    └── engine_data.csv       # Engine telemetry dataset
```
