# VEHICLE OBJECT DETECTION AND DATA COLLECTING
<p align='center'>
<img src="assets\main_banner.png"></img>
</p>

# OVERVIEW

The trained yolov8n model detects the vehicles based on the collected and labeled data. Each car has its own ID, which is attributed to the object from the first moment the object is detected by the model. 

After assigning an ID, the movement of its center is tracked, and depending on the change in its coordinates, the direction of movement of the car with a unique ID is determined. 

All received data is recorded in an xlsx file every certain time interval.

## PIPELINE
- [SUMMARY](#summary)
- [TRAINING SAMPLES COLLECTING](#training_samples)
- [AUGMENTATION](#augmentation)
- [MODEL TRAINING](#model_training)
- [DESIGNING CUSTOM UI](#ui)
- [VEHICLE TRACKING BY ID](#tracking)
- [RECEIVED DATA RECORDING](#data_recording)

### SUMMARY

<strong>TRAINING SAMPLES: 844</strong><br>
<strong>VALIDATION SAMPLES: 300</strong><br>
<strong>AUG TYPES: ROTATION, BRIGHTNESS AUGS</strong><br>
<strong>TOTAL TRAINING DATASET SIZE: 6088</strong><br>
<strong>MaP50: 0.78</strong><br>
<strong>MaP50-95: 0.53</strong><br>

<p align='center'>
<img src="assets\results.png"></img>
</p>

### TRAINING SAMPLES COLLECTING

The data was collected by obtaining video footage from public access traffic cameras in various cities, but mainly in Volgograd. Each video was divided into intervals in such a way as to get ~100 training samples per each unique video. During the training process, relevant data was added, which the model does not cope well with.

The validation dataset consists of 300 labeled night time and daytime samples, which use 30 different subsets of data from various locations(10 samples per subset).

The training dataset consists of 844 labeled data from 8 different videos.

<p align='center'>
<img src="assets\train_batch2.jpg"></img>
</p>

### AUGMENTATION

**Rotation augmentation** implements 4 types of rotations for each training sample: 30, -30, 45, -45 degrees of rotation.

The annotation coordinates for each object on the sample are changed by rotating the center of each bbox relative to the sample origin. Thus, all bboxes occupy the correct position on the augmented sample.

<p align='center'>
<img src="assets\rot_aug.png"></img>
</p>

**Brightness augmentation** implements 2 types of brightness augmentation for each training sample: with 0.5 and 1.5 factors.

(data_preparation.py consists aug code)

### DESIGNING CUSTOM UI

The interface is a bounding box and an ID assigned to this bounding box. The ID is indicated above the bounding box. ID tracking systems are described in the next paragraph.

Green box is ROI, where all predictions are happening. Model doesn't see anything beyond this box. 

<p align='center'>
<img src="assets\ui.gif"></img>
</p>

### VEHICLE TRACKING BY ID

With the new iteration j, the model iteratively goes through all predicted bounding boxes and checks whether the center of the i-th bounding box is located within the boundaries of any of the j-1 tracked bounding boxes.

In this case, a new ID is not created, and the data of the j-1 bounding box is overwritten with j bounding box data in tracking objects with the same ID. Otherwise, a new ID is created with the data of the current j bounding box.

Each tracked bounding box has its own lifetime. If it has not changed its center during some iterations, then the ID stops being tracked.

```py
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
```

### RECEIVED DATA RECORDING

The function writes data to an xlsx file every n seconds. Template for writing data: [year, month, day, hour, minute, second, roadname, orientation, direction1, direction2, startpoint(coordinates)]. 

Real example of recorded data for 15 seconds interval:

<p align='center'>
<img src="assets\video.gif"></img>
</p>

<p align='center'>
<img src="assets\data_recording.png"></img>
</p>

<p align='center'>
<img src="assets\map.png"></img>
</p>