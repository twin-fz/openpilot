#!/usr/bin/env python3
import time

from cereal import log, messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.realtime import set_realtime_priority
from openpilot.common.swaglog import cloudlog
from openpilot.frogpilot.common.frogpilot_variables import get_frogpilot_toggles, params_memory
from openpilot.frogpilot.system.yolo_detector import YoloDetector

DETECT_INTERVAL = 0.25  # 4 Hz default
THERMAL_THROTTLE_TEMP = 75.0  # Celsius
THERMAL_INTERVAL = 1.0  # 1 Hz when hot


def main():
  set_realtime_priority(1)
  cloudlog.warning("detectd: starting vision detection daemon")

  detector = YoloDetector()

  cloudlog.warning("detectd: connecting to road camera VisionIPC stream")
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not vipc_client.connect(False):
    time.sleep(0.2)
  cloudlog.warning(f"detectd: connected to road camera with buffer size: {vipc_client.buffer_len}")

  sm = messaging.SubMaster(["carState", "deviceState", "frogpilotPlan"])
  pm = messaging.PubMaster(["frogpilotDetections"])

  frogpilot_toggles = get_frogpilot_toggles()

  last_inference_time = 0.0

  while True:
    sm.update(0)

    if sm["frogpilotPlan"].togglesUpdated or params_memory.get_bool("FrogPilotTogglesUpdated"):
      frogpilot_toggles = get_frogpilot_toggles()

    # Demand-gated activation:
    red_light_needed = frogpilot_toggles.red_light_alert and (sm["frogpilotPlan"].redLight or sm["frogpilotPlan"].forcingStop)
    plate_needed = frogpilot_toggles.license_plate_detector and (sm["frogpilotPlan"].trackingLead or sm["carState"].standstill)
    active = red_light_needed or plate_needed

    now = time.monotonic()

    # Determine rate and thermal throttling
    cpu_temp = max(sm["deviceState"].cpuTempC) if len(sm["deviceState"].cpuTempC) > 0 else 50.0
    required_interval = THERMAL_INTERVAL if cpu_temp >= THERMAL_THROTTLE_TEMP else DETECT_INTERVAL

    if not active:
      # Send inactive state and sleep lightly to avoid CPU usage
      msg = messaging.new_message("frogpilotDetections")
      detections = msg.frogpilotDetections
      detections.frameId = vipc_client.frame_id if vipc_client else 0
      detections.timestamp = int(now * 1e9)
      detections.modelLoaded = detector.loaded
      detections.scanning = False
      detections.trafficLightState = log.FrogPilotDetections.TrafficLightState.none
      detections.trafficLightConfidence = 0.0
      detections.trafficLightBox = [0.0, 0.0, 0.0, 0.0]
      pm.send("frogpilotDetections", msg)

      time.sleep(0.05)
      continue

    # Active: enforce 4 Hz limit
    if now - last_inference_time < required_interval:
      time.sleep(0.02)
      continue

    buf = vipc_client.recv()
    if buf is None:
      time.sleep(0.01)
      continue

    last_inference_time = now

    # Run detection
    results = detector.detect(buf, detect_traffic_lights=red_light_needed, detect_plates=plate_needed)

    # Publish message
    msg = messaging.new_message("frogpilotDetections")
    detections = msg.frogpilotDetections
    detections.frameId = vipc_client.frame_id
    detections.timestamp = int(now * 1e9)
    detections.modelLoaded = detector.loaded
    detections.scanning = True

    # Traffic light state
    tl_state = results.get("traffic_light_state", "none")
    if tl_state == "red":
      detections.trafficLightState = log.FrogPilotDetections.TrafficLightState.red
    elif tl_state == "yellow":
      detections.trafficLightState = log.FrogPilotDetections.TrafficLightState.yellow
    elif tl_state == "green":
      detections.trafficLightState = log.FrogPilotDetections.TrafficLightState.green
    else:
      detections.trafficLightState = log.FrogPilotDetections.TrafficLightState.none

    detections.trafficLightConfidence = float(results.get("traffic_light_conf", 0.0))
    detections.trafficLightBox = list(results.get("traffic_light_box", [0.0, 0.0, 0.0, 0.0]))

    # License plates
    plates = results.get("license_plates", [])
    if len(plates) > 0:
      plate_list = detections.init("licensePlates", len(plates))
      for idx, plate in enumerate(plates):
        plate_list[idx].box = list(plate.get("box", [0.0, 0.0, 0.0, 0.0]))
        plate_list[idx].confidence = float(plate.get("confidence", 0.0))
        plate_list[idx].text = str(plate.get("text", ""))

    pm.send("frogpilotDetections", msg)


if __name__ == "__main__":
  main()
