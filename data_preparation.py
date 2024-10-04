#----------------------------CUT VIDEO EVERY X SECS
# import cv2
# import os

# video_link = r'data\part5.mkv'
# output_dir = r'data\training_samples_cutted_fullsize'
# file_names_without_extensions = []
# count = 0

# if len(os.listdir(output_dir)) != 0:

#     for name in os.listdir(output_dir):
#         file_names_without_extensions.append(int(name[0:-4]))

#     count = max(file_names_without_extensions) + 1

# cap = cv2.VideoCapture(video_link)
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# seconds = 5 # Set seconds here

# for frame in range(0, total_frames, int(fps*seconds)):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
#     ret, frame = cap.read()
    
#     if not ret:
#         break

#     file_name = os.path.join(output_dir, f'{count}.png')
#     cv2.imwrite(file_name, frame)
#     count += 1
    
# cap.release()

#----------------------------ROTATON AUGMENTATION
# import os
# import cv2
# import math
# import numpy as np

# def rotate_point(point, angle):
#     ox, oy = (320, 320)#origin
#     px, py = point
#     angle_rad = math.radians(-angle)

#     qx = int(ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy))
#     qy = int(oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy))
#     return (qx, qy)

# def yolo_to_pixel_coord(yolo_bbox):#bbox center to pixel coords
#   class_id, x_center, y_center, width, height = yolo_bbox
  
#   x_center_pixel = x_center * 640
#   y_center_pixel = y_center * 640
#   bbox_width_pixel = width * 640
#   bbox_height_pixel = height * 640
  
#   return [class_id, x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel]

# def duplicate_annotation(annotation, aug_type):
#   source_file = annotations_path + f'/{annotation}.txt'
#   destination_file = annotations_path + f'/aug_{aug_type}_{annotation}.txt'
  
#   with open(source_file, 'r') as src_file:
#     with open(destination_file, 'w') as dest_file:
      
#       for line in src_file:
#         dest_file.write(line)

# def rotate_img(img, angle):
#   rotation_matrix = cv2.getRotationMatrix2D((320, 320), angle, 1.0)
#   rotated_img = cv2.warpAffine(img, rotation_matrix, (640, 640))
  
#   return rotated_img

# def black_lined_aug(img_name, img, step, bar_width):
#   width, height, _ = img.shape

#   for i in range(0, width, step):
#     point1 = (i, 0)
#     point2 = (i + bar_width, height)
#     cv2.rectangle(img, point1, point2, (0, 0, 0), -1)
    
#   cv2.imwrite(images_path + f'/aug_lined_{img_name}.png', img)
#   duplicate_annotation(img_name, 'lined')
        
# def brightness_aug(img_name, img, factor):
#   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#   hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
#   bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  
#   cv2.imwrite(images_path + f'/aug_brightness_{factor}_{img_name}.png', bright_img)
#   duplicate_annotation(img_name, f'brightness_{factor}')

# def new_annotation(yolo_bbox, angle):
#   class_id, x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel = yolo_to_pixel_coord(yolo_bbox)#put center here
  
#   rotated_center = rotate_point((x_center_pixel, y_center_pixel), angle)
#   x_center = rotated_center[0] / 640
#   y_center = rotated_center[1] / 640
#   width = bbox_width_pixel / 640
#   height = bbox_height_pixel / 640
  
#   x_point1 = int(rotated_center[0] - bbox_width_pixel / 2)
#   y_point1 = int(rotated_center[1] - bbox_height_pixel / 2)
  
#   if x_point1 < 0 or y_point1 < 0 or x_point1 > 640 or y_point1 > 640:
#     return False
  
#   x_point2 = int(rotated_center[0] + bbox_width_pixel / 2)
#   y_point2 = int(rotated_center[1] + bbox_height_pixel / 2)
  
#   if x_point2 < 0 or y_point2 < 0 or x_point2 > 640 or y_point2 > 640:
#     return False
    
#   return [class_id, x_center, y_center, width, height]

# annotations_path = 'data/train/labels'
# images_path = 'data/train/images'
# angles = [30, -30, 45, -45]
# brightness_factors = [0.5, 1.5]

# for annotations in (os.listdir(annotations_path)):#all annotations for picture
#   img_filename = annotations[:-4]
#   img_path = images_path + f'/{img_filename}.png'
#   img = cv2.imread(img_path)
  
#   for angle in angles:
#     rotated_img = rotate_img(img, angle)
#     cv2.imwrite(images_path + f'/aug_{angle}_{img_filename}.png', rotated_img)
  
#     with open(annotations_path + f'/{annotations}', 'r') as file:#open first annotation_txt file
#       lines = file.readlines()#read all bboxes
#       annotation = []
      
#     for line in lines:
#       yolo_bbox = list(map(float, line.split()))#turn every bbox to a list
#       yolo_bbox[0] = int(yolo_bbox[0])
#       yolo_bbox_rotated = new_annotation(yolo_bbox, angle)
      
#       if yolo_bbox_rotated == False:
#         continue
      
#       annotation.append(yolo_bbox_rotated)
    
#     with open(annotations_path + f'/aug_{angle}_{annotations}', 'w') as filew:
      
#       for line in annotation:
#         filew.write(' '.join(map(str, line)) + '\n')
  
#   # black_lined_aug(img_filename, img.copy(), 15, 3)
  
#   for factor in brightness_factors:
#     brightness_aug(img_filename, img.copy(), factor)

#----------------------------VISUALIZE ANNOTATIONS ON PICTURES
import os
import cv2

filename = 'aug_45_092ce2ff-1685'
img = cv2.imread(f'data/train/images/{filename}.png')

with open(f'data/train/labels/{filename}.txt', 'r') as file:
  lines = file.readlines()

annotations = []
for line in lines:
    values = list(map(float, line.split()))  # Convert string to float
    annotations.append(values)

for i in annotations:
  x_center = i[1] * 640
  y_center = i[2] * 640
  width = i[3] * 640
  height = i[4] * 640
  
  point1 = (int(x_center - width / 2), int(y_center - height / 2))
  point2 = (int(x_center + width / 2), int(y_center + height / 2))
  
  cv2.rectangle(img, point1, point2, (0, 255, 0), 2)
  
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------RESTRICT THE IMAGE AND RESIZE TO 640X640
# import os
# import cv2

# img_input_path = r'data\training_samples_cutted_fullsize'
# img_output_path = r'data\training_samples_cutted_fullsize'

# for img_filename in (os.listdir(img_input_path)):
#   print(img_filename)
#   img = cv2.imread(img_input_path + f'\{img_filename}')
#   cv2.rectangle(img, (1300, 0), (1920, 372), (0, 0, 0), -1)
#   cv2.rectangle(img, (0, 0), (264, 364), (0, 0, 0), -1)
#   img_resized = cv2.resize(img, (640, 640))
#   cv2.imwrite(img_output_path + f'\{img_filename}', img_resized)