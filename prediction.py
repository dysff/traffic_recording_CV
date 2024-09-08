import cv2
from ultralytics import YOLO
import random
# import subprocess

test_link = 'image.png'
model = YOLO('yolov8n.pt')
VIDEO_PATH = 'vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4'

# streamlink_cmd = "streamlink https://www.twitch.tv/mobotixwebcamsrussia best --stream-url"
# stream_url = subprocess.check_output(streamlink_cmd, shell=True).decode("utf-8").strip()

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('YOLOv8 Real-Time Predictions', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('YOLOv8 Real-Time Predictions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
tracking_objects = {}

def blackout_top_bottom_third(image):
    height, width, _ = image.shape
    top_boundary = height // 3
    image[0:top_boundary, :] = 0
    
    return image

def update_ver_bboxes():
    keys_to_delete = [] # Create a list of keys to delete
    
    for obj in list(tracking_objects.keys()):
        
        if tracking_objects[obj][-1] > 150:
            keys_to_delete.append(obj)  # Add the key to the list for deletion
        
        else:
            tracking_objects[obj] = (tracking_objects[obj][0], tracking_objects[obj][1], 
                                     tracking_objects[obj][2], tracking_objects[obj][3], 
                                     tracking_objects[obj][4] + 1)

    for key in keys_to_delete:
        del tracking_objects[key]

def assign_id(bbox_center, new_x1, new_y1, new_x2, new_y2):
  update_ver_bboxes()
  
  for i in tracking_objects:
    x1 = tracking_objects[i][0]
    y1 = tracking_objects[i][1]
    x2 = tracking_objects[i][2]
    y2 = tracking_objects[i][3]
    
    if (x1 <= bbox_center[0] <= x2) and (y1 <= bbox_center[1] <= y2):
      tracking_objects[i] = (new_x1, new_y1, new_x2, new_y2, 1)
      return i
    
  new_id = random.randint(1, 99)
  tracking_objects[new_id] = (new_x1, new_y1, new_x2, new_y2, 1)
  return new_id

def id_plate(id_, img, x1, y1):
  plate_len = 190
  plate_wd = 48
  radius = plate_wd // 2
  
  if id_ < 10:
    plate_len = 150
  
  cv2.rectangle(img, (x1, y1 - plate_wd), (x1 + plate_len, y1), color=(232, 46, 195), thickness=-1)
  cv2.circle(img, (x1, y1 - radius), radius=radius, color=(232, 46, 195), thickness=-1)
  cv2.circle(img, (x1 + plate_len, y1 - radius), radius=radius, color=(232, 46, 195), thickness=-1)
  cv2.putText(img, f'{id_}:veh', (x1 - 10, y1 - 3), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)

def counter(img):
  height, width, _ = img.shape
  start_point = (50, height - int(height / 3))
  end_point = (width - 50, height - int(height / 3))
  
  cv2.line(img, start_point, end_point, (37, 194, 58), thickness=8)

def draw_UI(img, results): #bboxes, ids
  
    for result in results:
      
        for bbox in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in bbox.xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(232, 46, 195), thickness=8)

            bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # cv2.circle(img, bbox_center, radius=4, color=(0, 0, 255), thickness=-1) # bbox center

            coef = 4
            x1_ver_box = x1 + int((x2 - x1) / coef)
            y1_ver_box = y1 + int((y2 - y1) / coef)  
            
            x2_ver_box = x2 - int((x2 - x1) / coef)  
            y2_ver_box = y2 - int((y2 - y1) / coef)
            
            # draw id text
            id_ = assign_id(bbox_center, x1_ver_box, y1_ver_box, x2_ver_box, y2_ver_box)
            id_plate(id_, img, x1, y1)
            counter(img)
            
            # Draw the smaller verification box inside the original bounding box
            # cv2.rectangle(img, (x1_ver_box, y1_ver_box), (x2_ver_box, y2_ver_box), color=(0, 255, 0), thickness=2)

while True:
  ret, frame = cap.read()
  preprocessed_frame = blackout_top_bottom_third(frame.copy())
  
  if not ret or cv2.waitKey(1) == ord('q'):
        print('Error/Stream finished')
        cap.release()
        cv2.destroyAllWindows()
        break
  
  #Inner bounding boxes visulization
  # for i in tracking_objects.values():
  #   cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), color=(0, 255, 0), thickness=2)
    
  results = model(preprocessed_frame)
  draw_UI(frame, results)
  cv2.imshow('YOLOv8 Real-Time Predictions', frame)