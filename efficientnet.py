import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolov3 import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


class Efficientnet(object):
    _defaults = {
       
        "model_path"        : 'model_data/Efficientnet_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
       
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
      
        "input_shape"       : [640, 640],
       
        "phi"               : 2,
        
        "confidence"        : 0.5,
      
        "nms_iou"           : 0.3,
       
        "letterbox_image"   : False,
     
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

 
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    
    def generate(self):
        
        self.net    = YoloBody(self.anchors_mask, self.num_classes, phi = self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    
    def detect_image(self, image, crop = False, count = False):
     
        image_shape = np.array(np.shape(image)[0:2])
       
        image       = cvtColor(image)
        
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
       
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
       
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
           
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
           
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
          
                                           
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            print(top_boxes)
         
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] - 45).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        print(f"字体大小{font}")
       
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

       
        if count == True:
        
            print("top_label:", top_label)
            num=0
            for i in (top_label):
                if i ==1:
                    num+=1
         
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            
            if num == 0:
                radio=0
            else:
                radio=num/len(top_label)
       
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
           
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                
                crop_image = image.crop([left, top, right, bottom])
                
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".jpg"), quality=95, subsampling=0)
 
                print("save crop_" + str(i) + ".png to " + dir_save_path)
       
        s=0
        seed_image_list=[]
        germinate_image_list =[]
        seed_rectangle_width =[]
        seed_rectangle_length=[]
        seed_rectangle_area=[]
        seed_rectangle_perimeter=[]

        germinate_rectangle_width=[]
        germinate_rectangle_length=[]
        germinate_rectangle_area=[]
        germinate_rectangle_perimeter=[]

        coord = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)

          

            label = label.encode('utf-8')
            coord.append([np.int32((left+(right-left)/2)),np.int32((bottom-top)/2+top)])
           
          
                    
            if predicted_class == "germinate":
              
                  crop_image = image.crop([left, top, right, bottom])
               
                  germinate_image_list.append(crop_image)

                  s=s+1
           
            if predicted_class == "not germinate":
              
                  crop_image = image.crop([left, top, right, bottom])
               

                  seed_image_list.append(crop_image)
                  s=s+1
             

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
           

            if predicted_class == "not germinate":
                  for i in range(thickness):
                  
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(0, 255, 0))
           
                    rectangle_width=np.abs(right-left)
                    rectangle_length=np.abs(bottom-top)
                    rectangle_area=rectangle_width*rectangle_length
                    rectangle_perimeter=(rectangle_length+rectangle_width)*2
                    

                    seed_rectangle_width.append(rectangle_width)
                    seed_rectangle_length.append(rectangle_length)
                    seed_rectangle_area.append(rectangle_area)
                    seed_rectangle_perimeter.append(rectangle_perimeter)
            
                    
                  draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0,255,0))
                  draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            elif predicted_class == "germinate":
                 
                 for i in range(thickness):
              
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(0, 0, 255))
              
                    rectangle_width_germinate=np.abs(right-left)
                    rectangle_length_germinate=np.abs(bottom-top)
                    rectangle_area_germinate=rectangle_width*rectangle_length
                    rectangle_area_perimeter=(rectangle_length_germinate+rectangle_width_germinate)*2

                    germinate_rectangle_width.append(rectangle_width_germinate)
                    germinate_rectangle_length.append(rectangle_length_germinate)
                    germinate_rectangle_area.append(rectangle_area_germinate)
                    germinate_rectangle_perimeter.append(rectangle_area_perimeter)

                 draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0,0,255))
                 draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
              
            del draw

        return  image,radio,num, seed_image_list,germinate_image_list,len(top_label),seed_rectangle_width,seed_rectangle_length,\
                seed_rectangle_perimeter,seed_rectangle_area,germinate_rectangle_width,germinate_rectangle_length,germinate_rectangle_perimeter,germinate_rectangle_area,coord,top_boxes



    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
     
        image       = cvtColor(image)
        
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
       
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
           
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
           
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
             
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
               
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        
        image       = cvtColor(image)
       
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
      
        image       = cvtColor(image)
        
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
         
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
          
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
