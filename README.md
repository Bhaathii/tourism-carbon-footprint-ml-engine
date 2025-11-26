# 🌍 Tourism Carbon Footprint ML Engine

A machine learning-powered web application that predicts and analyzes carbon emissions from tourism activities. This project helps tourists and travel agencies understand their environmental impact and provides actionable recommendations for sustainable travel.

## 📋 Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- **Carbon Emissions Prediction**: Classifies trips into Low, Medium, or High emission categories
- **Comprehensive Input Parameters**: 13 different environmental factors
- **Personalized Recommendations**: Eco-friendly suggestions based on emission levels
- **Real-time Processing**: Fast ML predictions (< 100ms)
- **Web Interface**: User-friendly Streamlit dashboard

### Emission Categories
- **Trip Details**: Duration, distance, vehicle occupancy
- **Environmental Factors**: Congestion and terrain factors
- **Accommodation**: Electricity consumption and grid efficiency
- **Food & Waste**: Food emissions, waste, and plastic impact
- **Transport**: Direct transportation emissions

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Bhaathii/carbon-footprint-ml-engine.git
cd carbon-footprint-ml-engine
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

**Option 1: Web Interface (Recommended)**
```bash
cd code
streamlit run app.py
```
The app will open at `http://localhost:8501`

**Option 2: Command Line Prediction**
```bash
cd code
python predict.py
```

**Option 3: Train the Model**
```bash
cd code
python training.py
```

## 📁 Project Structure

```
carbon-footprint-ml-engine/
├── code/
│   ├── app.py                 # Streamlit web application
│   ├── predict.py             # Prediction script
│   ├── training.py            # Model training script
│   ├── recomendation.py       # Recommendation engine
│   └── __pycache__/
├── data/
│   └── tourism_5000_rows.csv  # Training dataset (5000 records)
├── model/
│   ├── model.pkl              # Trained Random Forest model
│   └── features.pkl           # Feature column names
├── README.md                  # This file
├── PROJECT_REPORT.md          # Detailed project documentation
└── GITHUB_SETUP.md            # GitHub setup guide
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9 |
| **Web Framework** | Streamlit |
| **Machine Learning** | scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Joblib |

## 📊 Dataset

The project uses a comprehensive tourism dataset with **5,000 records** containing:

**Features (13 input parameters):**
- `trip_days`: Trip duration (1-30 days)
- `distance_km`: Travel distance (1-1000 km)
- `occupancy`: Vehicle occupancy (1-100 passengers)
- `congestion_factor`: Traffic condition factor (0.8-1.2)
- `terrain_factor`: Road terrain factor (1.0-1.3)
- `nightly_kwh_est`: Nightly electricity consumption (1-200 kWh)
- `grid_ef_kgCO2_per_kWh`: Grid carbon efficiency (0.01-1.0)
- `food_emissions_kgCO2`: Food-related emissions (0-100 kgCO2)
- `waste_emissions_kgCO2`: Waste-related emissions (0-50 kgCO2)
- `plastic_emissions_kgCO2`: Plastic-related emissions (0-10 kgCO2)
- `transport_emissions_kgCO2`: Transport emissions (0-500 kgCO2)
- `accommodation_elec_kgCO2`: Electricity emissions (0-200 kgCO2)
- `accommodation_gen_kgCO2`: Generator emissions (0-200 kgCO2)

**Target Variable:**
- `emission_level`: Classification (Low, Medium, High)

## 🤖 Model Details

### Algorithm
- **Model Type**: Random Forest Classifier
- **Number of Trees**: 300
- **Training Data**: 5,000 tourism trips
- **Features**: 13 input parameters

### Performance Metrics
- **Overall Accuracy**: 90%
- **Precision (Medium)**: 89%
- **Recall (Medium)**: 100%
- **F1-Score (Medium)**: 94%

### Performance Optimization
- **Prediction Time**: < 100ms
- **Response Time**: < 1 second
- **Interface Load Time**: < 2 seconds

## 💻 Usage

### Web Application

1. Launch Streamlit app:
```bash
streamlit run code/app.py
```

2. Fill in the trip details:
   - Trip duration (days)
   - Travel distance (km)
   - Vehicle occupancy
   - Environmental factors (congestion, terrain)

3. Enter accommodation details:
   - Nightly electricity consumption
   - Grid carbon efficiency

4. Input food and waste information:
   - Food emissions
   - Waste emissions
   - Plastic emissions

5. Add transport details:
   - Transport emissions

6. Click "Predict Emission Level" to get:
   - Classification (Low/Medium/High)
   - Personalized eco-friendly recommendations

### Recommendations Logic

| Emission Level | Recommendation |
|---|---|
| **Low** | ✅ Excellent! Keep using eco-friendly transport, reusable items, and sustainable practices. |
| **Medium** | ⚠️ You can improve. Reduce meat consumption, lower electricity use, minimize plastic usage. |
| **High** | 🔴 High impact. Switch to public transport, reduce energy, eat vegetarian, avoid plastic, choose eco-hotels. |

## 📈 Model Training

To retrain the model with new data:

```bash
python code/training.py
```

This will:
1. Load the training data from `data/tourism_5000_rows.csv`
2. Train a new Random Forest model
3. Evaluate performance metrics
4. Save the model to `model/model.pkl`
5. Save feature columns to `model/features.pkl`

## 🔄 Workflow

```
User Input → Validation → Feature Processing → ML Model → 
Prediction → Recommendation Engine → Result Display
```

## 📚 File Documentation

### app.py
Main Streamlit web application with interactive dashboard and real-time predictions.

### predict.py
Standalone prediction script for command-line usage with sample data.

### training.py
Model training and evaluation script for the Random Forest classifier.

### recomendation.py
Recommendation engine for generating personalized eco-friendly suggestions.

## 🚀 Future Enhancements

- [ ] Database integration (PostgreSQL/MySQL)
- [ ] User accounts and trip history
- [ ] Comparison analytics between trips
- [ ] PDF report export
- [ ] Mobile application
- [ ] Real-time carbon tracking
- [ ] Integration with travel booking platforms
- [ ] API endpoint for third-party integrations

## 📄 Documentation

- **Full Project Report**: See `PROJECT_REPORT.md` for detailed architecture, diagrams, and implementation details
- **GitHub Setup Guide**: See `GITHUB_SETUP.md` for version control instructions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact & Support

For questions or support, please reach out through GitHub Issues.

---

**Made with ❤️ for sustainable tourism**