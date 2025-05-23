# Autonomous_Orchard_Monitoring

This project implements an object detection pipeline for fruit counting in orchards using a custom-trained YOLOv5 model. A drone captures video footage of an apple orchard, and the system detects apples in real time, aggregates their positions on screen sides, and generates a dynamic heatmap showing fruit density over time and space. Useful for yield estimation, harvest planning, and orchard productivity analysis.

Main features:
Real-time object detection using YOLOv5.
Spatial mapping of fruit detections (left/right frame sides).
Heatmap visualization correlating position and time.
Configurable confidence threshold and resolution.
