import os
import cv2 
from mtcnn import MTCNN


detector = MTCNN()
def xywh2xyxy(boxes):
    new_boxes = []
    for box in boxes:
        x,y,w,h = box
        new_box = [x,y,x+w,y+h]
        new_boxes.append(new_box)
    return new_boxes
def mtcnn_face_detect(frame,image_input = "bgr"):
    if image_input == "bgr":
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame)
    face = []
    if faces:
        confidence_scores = [face['confidence'] for face in faces]
        max_confidence_index = confidence_scores.index(max(confidence_scores))
        #x,y,w,h
        face = faces[max_confidence_index]['box']
    return xywh2xyxy([face])




input_folder  = r"D:\nam_code\dataset\dataset_origin"
output_folder = r"D:\nam_code\dataset\dataset_extracted"
os.makedirs(output_folder,exist_ok=True)

metadata_file = "classes.txt"
try:
    os.remove(metadata_file)
except:
    pass
for i, name in enumerate(os.listdir(input_folder)):
    with open(metadata_file,'a') as f:
        text = f"{i} {name}\n"
        f.write(text)
    sub_folder = os.path.join(input_folder,name)
    sub_output_folder   = os.path.join(output_folder,name)
    os.makedirs(sub_output_folder,exist_ok=True)
    for file_name in os.listdir(sub_folder):
        file_save = os.path.join(sub_output_folder,file_name)
        if os.path.exists(file_save):
            continue
        file_path = os.path.join(sub_folder,file_name)
        image = cv2.imread(file_path)
        face = mtcnn_face_detect(image)[0]
        x1,y1,x2,y2 = face
        face_image = image[y1:y2,x1:x2,:]
        file_save = os.path.join(sub_output_folder,file_name)
        cv2.imwrite(file_save,face_image)
