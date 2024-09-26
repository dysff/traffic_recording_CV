#----------------------------CUT VIDEO EVERY 3 SECS
# import cv2
# import os

# video_link = r'data\part2.mkv'
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

# seconds = 5

# for frame in range(0, total_frames, int(fps*seconds)): #every 3 secs
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

# def rotate_img(img, angle):
#   rotation_matrix = cv2.getRotationMatrix2D((320, 320), angle, 1.0)
#   rotated_img = cv2.warpAffine(img, rotation_matrix, (640, 640))
  
#   return rotated_img

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

# for annotations in (os.listdir(annotations_path)):#all annotations for picture
#   img_path = images_path + f'/{annotations[:-4]}.png'
#   img = cv2.imread(img_path)
  
#   for angle in angles:
#     rotated_img = rotate_img(img, angle)
#     cv2.imwrite(images_path + f'/aug_{angle}_{annotations[:-4]}.png', rotated_img)
  
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

#----------------------------RESIZE TO 640X640
# import os
# import cv2

# input_path = 'data/training_samples_fullsize'
# output_path = 'data/training_samples_resized'

# for image_name in (os.listdir(input_path)):
#   img = cv2.imread(input_path + f'/{image_name}')
#   resized_img = cv2.resize(img, (640, 640))
#   cv2.imwrite(output_path + f'/{image_name}', resized_img)

#----------------------------VISUALIZE ANNOTATIONS ON PICTURES
# import os
# import cv2

# img = cv2.imread(r'data\train\images\aug_45_f073860c-69.png')

# with open(r'data\train\labels\aug_45_f073860c-69.txt', 'r') as file:
#   lines = file.readlines()

# annotations = []
# for line in lines:
#     values = list(map(float, line.split()))  # Convert string to float
#     annotations.append(values)

# for i in annotations:
#   x_center = i[1] * 640
#   y_center = i[2] * 640
#   width = i[3] * 640
#   height = i[4] * 640
  
#   point1 = (int(x_center - width / 2), int(y_center - height / 2))
#   point2 = (int(x_center + width / 2), int(y_center + height / 2))
  
#   cv2.rectangle(img, point1, point2, (0, 255, 0), 2)
  
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------