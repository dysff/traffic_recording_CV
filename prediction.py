import cv2
from ultralytics import YOLO
import random
import subprocess

test_link = 'image.png'
model = YOLO('yolov8n.pt')
VIDEO_PATH = 'vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4'

# streamlink_cmd = "streamlink https://www.twitch.tv/mobotixwebcamsrussia best --stream-url"
# stream_url = subprocess.check_output(streamlink_cmd, shell=True).decode("utf-8").strip()

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('YOLOv8 Real-Time Predictions', cv2.WINDOW_NORMAL)
tracking_objects = {}

def assign_id(bbox_center, new_x1, new_y1, new_x2, new_y2):
  
  for i in tracking_objects:
    x1 = tracking_objects[i][0]
    y1 = tracking_objects[i][1]
    x2 = tracking_objects[i][2]
    y2 = tracking_objects[i][3]
    
    if (x1 <= bbox_center[0] <= x2) and (y1 <= bbox_center[1] <= y2):
      tracking_objects[i] = (new_x1, new_y1, new_x2, new_y2)
      return i
    
  new_id = random.randint(1, 1000)
  tracking_objects[new_id] = (new_x1, new_y1, new_x2, new_y2)
  return new_id

def draw_UI(img, results): #bboxes, ids
  
    for result in results:
      
        for bbox in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in bbox.xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

            bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # cv2.circle(img, bbox_center, radius=4, color=(0, 0, 255), thickness=-1)

            coef = 4
            x1_ver_box = x1 + int((x2 - x1) / coef)
            y1_ver_box = y1 + int((y2 - y1) / coef)  
            
            x2_ver_box = x2 - int((x2 - x1) / coef)  
            y2_ver_box = y2 - int((y2 - y1) / coef)
            
            id_ = assign_id(bbox_center, x1_ver_box, y1_ver_box, x2_ver_box, y2_ver_box)
            cv2.putText(frame, str(id_), bbox_center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)
            
            # Draw the smaller verification box inside the original bounding box
            cv2.rectangle(img, (x1_ver_box, y1_ver_box), (x2_ver_box, y2_ver_box), color=(0, 255, 0), thickness=2)

while True:
  ret, frame = cap.read()
  
  if not ret or cv2.waitKey(1) == ord('q'):
        print('Error/Stream finished')
        cap.release()
        cv2.destroyAllWindows()
        break
    
  results = model(frame)
  draw_UI(frame, results)
  cv2.imshow('YOLOv8 Real-Time Predictions', frame)