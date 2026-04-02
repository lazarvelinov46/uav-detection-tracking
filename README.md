# UAV Detection and Tracking

PhD-level deep learning project for detecting and tracking UAVs in video sequences.

## Architecture
- **Detector:** YOLOv8 (anchor-free, single-stage) — YOLOX as fallback
- **Tracker:** ByteTrack (primary) / DeepSORT (comparison baseline)
- **Metrics:** MOTA, IDF1, HOTA

## Datasets
- Anti-UAV v4 (RGB + Thermal IR)
- Drone-vs-Bird
- Secondary: VisioDECT, MAV-VID, UAVDT

## Setup
See environment setup instructions below (Step 4 onwards).

## Project Status
🔧 Phase 1: Environment Setup & Data Preparation
