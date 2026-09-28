#!/usr/bin/env python3
import math
import os
from pathlib import Path
from typing import Optional

try:
  import numpy as np
except ImportError:
  np = None

try:
  import onnxruntime as ort
except ImportError:
  ort = None

from openpilot.common.swaglog import cloudlog

DEFAULT_MODEL_PATH = Path("/data/models/detect_yolo.onnx")
INPUT_SIZE = 416
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Class indices for custom multi-class detection model
# 0: red light, 1: yellow light, 2: green light, 3: license plate
CLASS_RED_LIGHT = 0
CLASS_YELLOW_LIGHT = 1
CLASS_GREEN_LIGHT = 2
CLASS_LICENSE_PLATE = 3


class YoloDetector:
  """
  Multi-class vision detector for traffic lights and license plates.
  Runs an optimized ONNX YOLO model (e.g. YOLOv8n) at 3-5 Hz.
  """
  def __init__(self, model_path: Optional[Path] = None):
    self.model_path = model_path or DEFAULT_MODEL_PATH
    self.session = None
    self.input_name = None
    self.input_shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    self.loaded = False

    self._init_session()

  def _init_session(self):
    if ort is None or np is None:
      cloudlog.info("detectd: onnxruntime or numpy not available, detector in stub mode")
      return

    if not self.model_path.is_file():
      cloudlog.info(f"detectd: model not found at {self.model_path}, detector waiting for model")
      return

    try:
      options = ort.SessionOptions()
      options.intra_op_num_threads = 2
      options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
      options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

      providers = ["CPUExecutionProvider"]
      if "QNNExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "QNNExecutionProvider")

      self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=providers)
      self.input_name = self.session.get_inputs()[0].name
      self.input_shape = self.session.get_inputs()[0].shape
      self.loaded = True
      cloudlog.warning(f"detectd: YOLO model successfully loaded from {self.model_path}")
    except Exception as e:
      cloudlog.error(f"detectd: failed to load ONNX model: {e}")
      self.loaded = False

  def preprocess(self, buf) -> Optional[np.ndarray]:
    """
    Extracts RGB image from VisionBuf (NV12 format) and resizes to model input.
    """
    if np is None:
      return None

    try:
      # NV12 layout: Y plane followed by UV plane
      h = buf.height
      w = buf.width
      stride = buf.stride

      # Map buffer
      raw = np.frombuffer(buf.data, dtype=np.uint8)
      y_plane = raw[:h * stride].reshape((h, stride))[:, :w]

      # Extract center region of interest (road corridor: upper 70% to avoid dashboard)
      crop_y2 = int(h * 0.85)
      y_crop = y_plane[:crop_y2, :]

      # Resample / resize to INPUT_SIZE x INPUT_SIZE
      scale_y = y_crop.shape[0] / INPUT_SIZE
      scale_x = y_crop.shape[1] / INPUT_SIZE
      yi = (np.arange(INPUT_SIZE) * scale_y).astype(int)
      xi = (np.arange(INPUT_SIZE) * scale_x).astype(int)
      sampled = y_crop[np.ix_(yi, xi)]

      # Convert grayscale Y to 3-channel normalized float32 tensor [1, 3, H, W]
      rgb = np.stack([sampled, sampled, sampled], axis=0).astype(np.float32) / 255.0
      return np.expand_dims(rgb, axis=0)
    except Exception as e:
      cloudlog.error(f"detectd preprocess error: {e}")
      return None

  def detect(self, buf, detect_traffic_lights: bool = True, detect_plates: bool = False) -> dict:
    result = {
      "traffic_light_state": "none",
      "traffic_light_conf": 0.0,
      "traffic_light_box": [0.0, 0.0, 0.0, 0.0],
      "license_plates": []
    }

    if not self.loaded or self.session is None:
      return result

    tensor = self.preprocess(buf)
    if tensor is None:
      return result

    try:
      outputs = self.session.run(None, {self.input_name: tensor})
      predictions = outputs[0]  # YOLO format: [1, 4 + classes, num_boxes]
      result = self._parse_predictions(predictions, detect_traffic_lights, detect_plates)
    except Exception as e:
      cloudlog.error(f"detectd inference error: {e}")

    return result

  def _parse_predictions(self, output: np.ndarray, check_lights: bool, check_plates: bool) -> dict:
    """
    Parses raw YOLO output boxes and confidences.
    """
    result = {
      "traffic_light_state": "none",
      "traffic_light_conf": 0.0,
      "traffic_light_box": [0.0, 0.0, 0.0, 0.0],
      "license_plates": []
    }

    if output.ndim == 3 and output.shape[1] > output.shape[2]:
      output = output.transpose(0, 2, 1)

    preds = output[0]  # [num_boxes, 4 + num_classes]
    num_classes = preds.shape[1] - 4

    boxes = preds[:, :4]  # xc, yc, w, h
    scores = preds[:, 4:]

    best_light_conf = 0.0
    best_light_state = "none"
    best_light_box = [0.0, 0.0, 0.0, 0.0]

    for i in range(preds.shape[0]):
      class_id = int(np.argmax(scores[i]))
      conf = float(scores[i, class_id])

      if conf < CONF_THRESHOLD:
        continue

      xc, yc, w, h = boxes[i]
      norm_box = [float(xc / INPUT_SIZE), float(yc / INPUT_SIZE), float(w / INPUT_SIZE), float(h / INPUT_SIZE)]

      # Traffic light detection
      if check_lights and class_id in (CLASS_RED_LIGHT, CLASS_YELLOW_LIGHT, CLASS_GREEN_LIGHT):
        if conf > best_light_conf:
          best_light_conf = conf
          best_light_box = norm_box
          if class_id == CLASS_RED_LIGHT:
            best_light_state = "red"
          elif class_id == CLASS_YELLOW_LIGHT:
            best_light_state = "yellow"
          elif class_id == CLASS_GREEN_LIGHT:
            best_light_state = "green"

      # License plate detection
      if check_plates and class_id == CLASS_LICENSE_PLATE:
        result["license_plates"].append({
          "box": norm_box,
          "confidence": conf,
          "text": ""
        })

    result["traffic_light_state"] = best_light_state
    result["traffic_light_conf"] = best_light_conf
    result["traffic_light_box"] = best_light_box
    return result
