# Hybrid_GNN_based_Earthquake_Alert_System

**📖 Project Overview:**

I built an advanced AI system that can forecast earthquake activity by combining time patterns and geographical relationships. This is one of the first successful attempts to use hybrid deep learning for earthquake prediction.

**🎯 What Problem This Solves:**

1. Earthquakes are hard to predict because they depend on both:

2. Time patterns (when earthquakes happen)

3. Space patterns (where earthquakes happen)

4. Traditional methods look at only one of these. My system looks at both together.

**What is Built here:**

Phase 1: Data Foundation:

1. Collected 112,099 earthquakes from 2010-2024

2. Cleaned and prepared the data

3. Grouped earthquakes into 15 tectonic regions using AI clustering

Phase 2: Basic AI Models:

1. Random Forest: 72% accurate at predicting earthquake magnitude

2. Gradient Boosting: Learned complex earthquake patterns

Phase 3 - Advanced AI:

1. LSTM Network: Learned time patterns in earthquake sequences

2. Graph Neural Networks: Learned how different tectonic regions connect

Phase 4 - Research Breakthrough:

1. Hybrid Model: Combined time and space learning

2. Result: 89% accurate at forecasting seismic activity

**Key Achievements:**

Performance Results:

1. 89.0% Prediction Accuracy.

2. 93.9% F1-Score (balance of precision and recall).

3. 99.4% Precision (almost never wrong when predicting earthquakes).

4. Good Reproducibility (same results every time).

**Technical Innovations:**

1. First Hybrid Architecture for earthquake forecasting.

2. Memory-Optimized Training for large datasets.

3. Geological Knowledge Integration with AI.

4. Complete Research Pipeline from raw data to trained model.

**How It Works:**

1. Data Collection:

   Source - USGS Earthquake Catalog.

   Period: 2010-2024.

   Earthquakes: 112,099 events (Magnitude 4.5+).

2. Data Processing:
    1. python.
    2. what happens:
        1. Clean earthquake data.
        2. Group into tectonic regions.
        3. Create daily activity features.
        4. Build time sequences (30 days).
           
3. AI Model Architecture:
    1. text.
    2. Input → [LSTM Time Learning] + [Regional Space Learning] → Combined → Prediction.
   
4. Training & Evaluation:
    1. Train on 80% of data.
    2. Test on 20% of data.
    3. Multiple validation runs.

5. Comprehensive metrics:
