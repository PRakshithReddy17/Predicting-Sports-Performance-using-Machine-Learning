# Predicting Sports Performance using Machine Learning

A comprehensive machine learning project for analyzing and predicting sports player performance using real-time monitoring and data visualization.

## Overview

This project implements an interactive Streamlit web application that explores IPL (Indian Premier League) 2025 player dataset with multiple visualization techniques and machine learning models for performance prediction.

## Features

- **Interactive Dashboard**: Real-time monitoring of sports analytics with Streamlit
- **Multiple Visualizations**:
  - Scatter plots, bar charts, line charts
  - Histograms, box plots, violin plots
  - Density plots (KDE), pair plots
  - Correlation heatmaps
  - Radar charts for player profile analysis
- **Machine Learning Models**: Random Forest regression with cross-validation
- **Performance Metrics**: MAE, RMSE, and R² score calculation
- **Data Export**: Download data as CSV or Excel format

## Dataset

The project analyzes 12 IPL players with the following metrics:
- **Player**: Player name
- **Matches**: Number of matches played
- **Runs**: Total runs scored
- **Average**: Batting average
- **StrikeRate**: Strike rate percentage
- **Fours**: Number of fours hit
- **Sixes**: Number of sixes hit

Players included: Sai Sudharsan, Shubman Gill, Suryakumar Yadav, Virat Kohli, Ruturaj Gaikwad, KL Rahul, Sanju Samson, Rinku Singh, Heinrich Klaasen, Nicholas Pooran, Travis Head, Abhishek Sharma

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- openpyxl

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRakshithReddy17/Predicting-Sports-Performance-using-Machine-Learning.git
cd Predicting-Sports-Performance-using-Machine-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run dk.py
```

The web app will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Plot Type**: Choose from the sidebar dropdown to switch between different visualization types
2. **Configure Axes**: Select X and Y axes columns for scatter plots and other visualizations
3. **Adjust Parameters**: Use sliders to modify bins for histograms and select top N players
4. **View Data**: Toggle the raw data table to see the dataset
5. **Train Models**: Use the machine learning section to train RandomForest models and view predictions
6. **Export Data**: Download the dataset as CSV or Excel

## Project Structure

```
.
├── dk.py                    # Main Streamlit application
├── ml_project.ipynb         # Jupyter notebook with analysis
├── README.md                # This file
├── LICENSE                  # MIT License
└── requirements.txt         # Python dependencies
```

## Machine Learning Approach

- **Model**: Random Forest Regressor (200 estimators)
- **Cross-Validation**: 5-fold KFold with shuffling
- **Features**: Selectable from Matches, Average, StrikeRate, Fours, Sixes
- **Target**: Player Runs
- **Evaluation**: MAE, RMSE, R² score with Actual vs Predicted visualization

## Performance Tips

- With only 12 rows of data, this is illustrative and not production-ready
- Use cross-validation to avoid overfitting on small datasets
- Experiment with different feature combinations for optimal results
- Hover over charts to explore player-level insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

PRakshithReddy17

## Disclaimer

This project is for educational purposes. The IPL data is illustrative and intended for demonstration of machine learning and data visualization techniques.
