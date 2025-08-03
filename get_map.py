import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolov5 import YOLOv5
from centernet import CenterNet
from yolox import YOLOX
from GHOSTyolov5 import GHOSTyolov5
from efficientnet import Efficientnet
from frcnn import FRCNN

if __name__ == "__main__":
    
    Out_dir = ['map_out_yolov5','map_out_GHOST_yolov5','map_out_yolox','map_out_CenterNet',"map_out_Faster-R-CNN","map_out_Efficientnet"]
    #"-> yolov5:0, GHOST_yolov5:1, yolox:2, CenterNet:3, Faster R-CNN:4, Efficientnet:5"
    model_list = [YOLOv5,GHOSTyolov5,YOLOX,CenterNet,FRCNN,Efficientnet]
    #"-> yolov5:0, GHOST_yolov5:1, yolox:2, CenterNet:3, Faster R-CNN:4, Efficientnet:5"
   
    map_mode        = 0
    
    classes_path    = 'model_data/voc_classes.txt'
  
    MINOVERLAP      = 0.5
    
    confidence      = 0.001
  
    nms_iou         = 0.5
   
    score_threhold  = 0.5
   
    map_vis         = False
  
    Test1_path  = 'Test1_data'
   
    map_out_path    = Out_dir[3]   #"< --------------- select"
    
    image_ids = open(os.path.join(Test1_path, "ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        model = model_list[3](confidence = confidence, nms_iou = nms_iou) #"<----------select"
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(Test1_path, "JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            model.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(Test1_path, "Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
