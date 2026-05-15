"""Auxiliary perception models (object detection, future: depth)."""

from .yolo_detector import YoloDetection, YoloDetector

__all__ = ["YoloDetector", "YoloDetection"]
