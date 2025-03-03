# Football Analysis System

## Overview
This Football Analysis System is an advanced sports analytics platform that combines computer vision, tracking algorithms, and statistical analysis to provide comprehensive insights into football (soccer) matches. The system processes video feeds to track players and the ball, analyze movements, detect key events, and present data through an interactive dashboard.

## Features
- **Player and Ball Detection**: Utilizes YOLOv8 for accurate detection
- **Player Tracking**: Implements DeepSORT for consistent player tracking across frames
- **Pose Estimation**: Analyzes player movements and postures
- **Event Detection**: Automatically identifies key events like passes, shots, and tackles
- **Statistical Analysis**: Processes raw data into actionable insights
- **Real-time Processing**: Handles live video feeds with minimal latency
- **Interactive Dashboard**: Visualizes data through a Streamlit interface

## System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- Minimum 8GB RAM (16GB recommended)
- 50GB available storage space

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/football-analysis-system.git
cd football-analysis-system
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download pre-trained models
```bash
python scripts/download_models.py
```

## Usage

### Configuration
Before running the system, configure the parameters in `config/system_config.yaml`:

```yaml
# Example configuration
input:
  source_type: "video"  # Options: "video", "rtsp", "webcam"
  source_path: "data/match.mp4"  # For video files
  # source_path: "rtsp://example.com/stream"  # For RTSP streams
  
detection:
  model: "yolov8m.pt"
  confidence: 0.5
  
tracking:
  max_age: 30
  min_hits: 3
  
analysis:
  teams:
    home: "Team A"
    away: "Team B"
  field_dimensions:
    width: 105  # meters
    height: 68  # meters
```

### Running the System

#### 1. Process a recorded video
```bash
python run_analysis.py --config config/system_config.yaml
```

#### 2. Process a live feed
```bash
python run_analysis.py --config config/system_config.yaml --real-time
```

#### 3. Launch the dashboard
```bash
python dashboard/app.py
```
Access the dashboard at `http://localhost:8501`

## Dashboard Features
The Streamlit dashboard provides:
- Live match statistics
- Player heatmaps and movement analysis
- Event timeline
- Team formation visualization
- Performance metrics comparison
- Video playback with synchronized analytics

## File Structure
```
football-analysis-system/
├── config/
│   └── system_config.yaml
├── data/
│   └── sample_videos/
├── models/
│   ├── detection/
│   ├── tracking/
│   └── pose/
├── src/
│   ├── detection.py
│   ├── tracking.py
│   ├── pose_estimation.py
│   ├── event_detection.py
│   ├── statistics.py
│   └── real_time.py
├── dashboard/
│   ├── app.py
│   └── components/
├── scripts/
│   └── download_models.py
├── utils/
│   └── visualization.py
├── run_analysis.py
└── requirements.txt
```

## Output Data
The system generates analysis results in the `output/` directory:
- Processed video with annotations
- JSON files with tracking data
- CSV files with event and statistical data
- Visualization images and charts

## Extending the System
The modular architecture allows for easy extension:
- Add new detection models in `src/detection.py`
- Implement custom event detectors in `src/event_detection.py`
- Create new dashboard visualizations in `dashboard/components/`

## Troubleshooting
- **GPU Memory Issues**: Reduce batch size and model size in configuration
- **Tracking Failures**: Adjust tracking parameters (max_age, min_hits)
- **Real-time Performance**: Lower video resolution or processing framerate

## License
[Specify your license here]

## Acknowledgments
- YOLOv8 by Ultralytics
- DeepSORT algorithm
- Streamlit visualization library
