import cv2
from ultralytics import YOLO
from random import choice
import time
from datetime import datetime
import pandas as pd

model = YOLO(r'runs\detect\train4\weights\best.pt')
model.to('cuda')
VIDEO_PATH = r'data\testing_videos\2024-10-01 15-11-42 (online-video-cutter.com).mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('YOLOv8 Real-Time Predictions', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('YOLOv8 Real-Time Predictions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

tracking_objects = {}
position_data = {} # {id: [center_past, center_current]}
counted_ids = []
cars_moving_upward = 0
cars_moving_downward = 0
cars_moving_left = 0
cars_moving_right = 0

start_time = time.time()
record_interval = 15
file_name = 'traffic_data.xlsx'

def record_to_excel(roadname, orientation, startpoint, switch):
  global cars_moving_upward, cars_moving_downward, cars_moving_left, cars_moving_right
  current_time = datetime.now()
  
  if switch:
    direction_1 = cars_moving_upward
    direction_2 = cars_moving_downward
    
  else:
    direction_1 = cars_moving_left
    direction_2 = cars_moving_right
    
  data = {
    'Year': [2024],
    'Month': [current_time.month],
    'Day': [current_time.day],
    'Hours': [current_time.hour],
    'Minutes': [current_time.minute],
    'Seconds': [current_time.second],
    'Roadname': [roadname],
    'Orientation': [orientation],
    'Direction1': [direction_1],
    'Direction2': [direction_2],
    'Startpoint': [startpoint]
    }
  
  df = pd.DataFrame(data)
  reader = pd.read_excel(file_name, engine='openpyxl')
  writer = pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay')
  df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
  writer.close()

def predicting_area(x_top, y_top, x_down, y_down, img):
  """
  Restricted area on the picture, where
  model is predicting objects(it doesn't see any objects beyond this area)
  
  Parameters
  ----------
  x_top : x of the upper-left corner of the box
  y_top : y of the upper-left corner of the box
  x_down : x of the lower-right corner of the box
  y_down : y of the lower-right corner of the box
  img : image
  """
  height, width, _ = img.shape
  
  cv2.rectangle(img, (0, 0), (width, y_top), (0, 0, 0), -1)#upper area
  cv2.rectangle(img, (0, y_top), (x_top, y_down), (0, 0, 0), -1)#left area
  cv2.rectangle(img, (0, y_down), (width, height), (0, 0, 0), -1)#lower area
  cv2.rectangle(img, (x_down, y_top), (width, y_down), (0, 0, 0), -1)#right area
  
  return img
  
def update_ver_bboxes():
  keys_to_delete = []
  
  for obj in list(tracking_objects.keys()):
      
    if tracking_objects[obj][-1] > 35:
      keys_to_delete.append(obj)
    
    else:
      tracking_objects[obj] = (tracking_objects[obj][0], tracking_objects[obj][1], 
                                tracking_objects[obj][2], tracking_objects[obj][3], 
                                tracking_objects[obj][4] + 1)

  for key in keys_to_delete:
    del tracking_objects[key]
    del position_data[key]
    
    if key in counted_ids:
      counted_ids.remove(key)

def assign_id(bbox_center, new_x1, new_y1, new_x2, new_y2):
  """
  Parameters
  ----------
  bbox_center : current bbox center tuple(x, y)
  new_x1 : x of the upper-left corner of the bbox
  new_y1 : y of the upper-left corner of the bbox
  new_x2 : x of the lower-right corner of the bbox
  new_y2 : y of the lower-right corner of the bbox
  """
  update_ver_bboxes()#Delete untracked IDs
  
  for i in tracking_objects:
    x1 = tracking_objects[i][0]
    y1 = tracking_objects[i][1]
    x2 = tracking_objects[i][2]
    y2 = tracking_objects[i][3]
        
    if (x1 <= bbox_center[0] <= x2) and (y1 <= bbox_center[1] <= y2):#Check if bbox center is located within the boundaries
      tracking_objects[i] = (new_x1, new_y1, new_x2, new_y2, 1)
      return i
    
  new_id = choice([i for i in range(1, 100) if i not in tracking_objects.keys()])#Create new random ID
  tracking_objects[new_id] = (new_x1, new_y1, new_x2, new_y2, 1)#Track new ID
  return new_id

def id_plate(id_, img, x1, y1):
  plate_len = 90
  plate_wd = 28
  radius = int(plate_wd / 2)
  
  if id_ < 10:
    plate_len = 70
  
  cv2.rectangle(img, (x1, y1 - plate_wd), (x1 + plate_len, y1), color=(232, 46, 195), thickness=-1)
  cv2.circle(img, (x1, y1 - radius), radius=radius, color=(232, 46, 195), thickness=-1)
  cv2.circle(img, (x1 + plate_len, y1 - radius), radius=radius, color=(232, 46, 195), thickness=-1)
  cv2.putText(img, f'{id_}:veh', (x1 - 10, y1 - 3), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

def traffic_counter(switch):
  global cars_moving_upward, cars_moving_downward, cars_moving_left, cars_moving_right
    
  for i in tracking_objects:
    
    if i not in counted_ids:
      past_x = position_data[i][0][0]
      current_x = position_data[i][1][0]
      past_y = position_data[i][0][1]
      current_y = position_data[i][1][1]
          
      if past_y != current_y:
        counted_ids.append(i)
        
        if switch: #if switch is 1, then it's up-down, if 0, then it's left-right  
          if past_y > current_y: # Moving upward
            cars_moving_upward += 1
                    
          elif past_y < current_y: # Moving downward
            cars_moving_downward += 1
        
        else:
          if past_x > current_x: # Moving left
            cars_moving_left += 1
            
          elif past_x < current_x: # Moving right
            cars_moving_right += 1
          
def traffic_count_plate(img, switch):
  cv2.rectangle(img, (100, 100 - 28), (500, 160), color=(232, 46, 195), thickness=-1)
  
  if switch:
    cv2.putText(img, f'Downward cars: {cars_moving_downward}', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Upward cars: {cars_moving_upward}', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

  else:
    cv2.putText(img, f'Leftward cars: {cars_moving_left}', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Rightward cars: {cars_moving_right}', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

def draw_UI(img, x_top, y_top, x_down, y_down, results, switch):
  cv2.rectangle(img, (x_top, y_top), (x_down, y_down), (0, 255, 0), 3)
  
  for result in results:
    
    for bbox in result.boxes:
      x1, y1, x2, y2 = [int(i) for i in bbox.xyxy[0]]
      cv2.rectangle(img, (x1, y1), (x2, y2), color=(232, 46, 195), thickness=6)

      bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
      # cv2.circle(img, bbox_center, radius=4, color=(0, 0, 255), thickness=-1) # bbox center

      coef = 8
      x1_ver_box = x1 + int((x2 - x1) / coef)
      y1_ver_box = y1 + int((y2 - y1) / coef)
      
      x2_ver_box = x2 - int((x2 - x1) / coef)  
      y2_ver_box = y2 - int((y2 - y1) / coef)
      
      id_ = assign_id(bbox_center, x1_ver_box, y1_ver_box, x2_ver_box, y2_ver_box)
      
      # update past and current bbox center position
      if id_ not in position_data:
        position_data[id_] = [bbox_center, bbox_center]
        
      else:
        position_data[id_] = [position_data[id_][1], bbox_center]
        
      id_plate(id_, img, x1, y1)
      traffic_counter(switch)
    
    traffic_count_plate(img, switch)

while True:
  ret, frame = cap.read()
  
  pred_area_top_x = 1000
  pred_area_top_y = 450
  pred_area_down_x = 1613
  pred_area_down_y = 700
  switch = 1 # True: up-down, False: left-right
  
  preproccessed_img = predicting_area(pred_area_top_x, pred_area_top_y, pred_area_down_x, pred_area_down_y, frame.copy())
  
  if not ret or cv2.waitKey(1) == ord('q'):
        print('Error/Stream finished')
        cap.release()
        cv2.destroyAllWindows()
        break
  
  #Inner id bounding boxes visulization
  for i in tracking_objects.values():
    cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), color=(0, 255, 0), thickness=2)
    
  results = model(preproccessed_img)
  draw_UI(frame, pred_area_top_x, pred_area_top_y, pred_area_down_x, pred_area_down_y, results, switch)
  cv2.imshow('YOLOv8 Real-Time Predictions', frame)
  
  if time.time() - start_time >= record_interval:
    record_to_excel('karelskaya st.', 's-w/n-e', '61.703348136600276, 30.691587139403435', switch)
    cars_moving_downward = 0
    cars_moving_upward = 0
    start_time = time.time()