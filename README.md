# UAV Detection and Tracking

PhD-level deep learning project for detecting and tracking UAVs in video sequences.

## Architecture
- **Detector:** YOLOv8 (anchor-free, single-stage), YOLOX
- **Tracker:** ByteTrack (primary) / DeepSORT (comparison baseline)
- **Metrics:** MOTA, IDF1, HOTA; for TrackEval git clone https://github.com/JonathonLuiten/TrackEval

## Datasets
- Anti-UAV v4 (Thermal IR)
