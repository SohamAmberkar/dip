# Project-ASES: Eye-Tracking Engagement Detection System

This project detects user engagement using eye-tracking and facial landmarks. It uses OpenCV, dlib, and Streamlit to process video input, determine engagement levels, and display results on a web-based interface.

## Features
- **Real-time Video Stream**: Detects engagement based on the Eye Aspect Ratio (EAR).
- **Automatic Screenshot Capture**: Saves images when disengagement is detected.
- **Engagement Report**: Logs EAR values and engagement status in a CSV file.
- **Web Interface with Streamlit**: View live video, saved screenshots, and reports.

## Installation

### Prerequisites
- Python 3.x
- Git
- Required Python libraries

### Clone the Repository
```bash
git clone https://github.com/SohamAmberkar/dip.git
cd Project-ASES
```
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the program
```bash
streamlit run streamlit_app.py

