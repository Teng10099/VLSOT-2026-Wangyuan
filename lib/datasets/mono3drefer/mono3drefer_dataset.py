# Release staging 2026-10-23: private path comments removed; runtime logic unchanged.
"""
Mono3DVG (https://github.com/ZhanYang-nwpu/Mono3DVG)
@ Paper: https://arxiv.org/pdf/2312.08022.pdf
@ Dataset: https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link
"""

import os
import ast
import tqdm,json
import numpy as np
import re
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re

from lib.datasets.utils import angle2class
from lib.datasets.mono3drefer.mono3drefer_utils import get_objects_from_label,get_info_from_json
from lib.datasets.mono3drefer.mono3drefer_utils import Calibration
from lib.datasets.mono3drefer.mono3drefer_utils import get_affine_transform,affine_scale_only
from lib.datasets.mono3drefer.mono3drefer_utils import affine_transform
from .pd import PhotometricDistort
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer

class Mono3DRefer_Dataset(data.Dataset):
    def __init__(self, split, cfg, form,bert_model='bert-base-uncased'):
## 
        
        # basic configuration
        self.root_dir = cfg.get('root_dir') # ropp3d的
        ###
        self.anno_json = cfg.get('anno_json')
        self.calib_dir = cfg.get('calib_path')
        self.img_root_dir = cfg.get('root_img_dir')
        self.video_dir = cfg.get('video_dir')
        self.id2style = {
            1: 'YOUTH',
            2: 'ELDER',
            3: 'POLICE',
            4: 'ENGINEER',
            5: 'CHILD',
            6: 'TEACHER'
        }
        self.Rcooper_video = cfg.get('Rcooper_vedio')
        self.Rope3D_video = cfg.get('Rope3D_vedio')
        ###
        self.split = split
        self.max_objs = 1
        self.lstm = False
        self.query_len = 120
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        self.cls2id = {'CAR':0,"TRUCK":1,"BUS":2,"VAN":3}#} class name to id   # class name to id
        self.class_name = len(self.cls2id)
        self.resolution = np.array([1920, 1080])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)

        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)

        # data split loading
        assert self.split in ['train', 'val', 'test']
        
        self.split_file1 = os.path.join(self.root_dir, 'Rcooper',self.split + '.txt')
        self.split_file2 = os.path.join(self.root_dir, 'Rope3D',self.split + '.txt')
        # 根据划分的训练集的视频编号来读取
        self.videos_list1 = [x.strip()+'.json' for x in open(self.split_file1).readlines()]
        self.videos_list2 = [x.strip()+'.json' for x in open(self.split_file2).readlines()]
        # load annotations for images
        self.anno_data = []
        if self.split == 'train':
            if form in ['1','3']:
                for _ in tqdm.tqdm(range(len(self.videos_list1)),desc=f"Loading {self.split} videos"):
                    video = self.videos_list1[_]
                    with open(os.path.join(self.Rcooper_video, video), 'r') as f:
                        anno = json.load(f)
                        #要加的属性
                        style_list = list(anno['description'].keys())
                        for i in range(len(style_list)):
                            videoID = 'Rcooper'+anno['videoID']+self.id2style[i+1]
                            # sequence_id = anno['sequence_id']
                            track_id = anno['track_id']
                            color = anno['color']
                            state = anno['state']
                            car_type = anno['type'].upper()
                            description = anno['description'][self.id2style[i+1]]
                            videos = [anno[f"frame{i}"]  for i in range(30)]
                            self.anno_data.append((videoID, track_id, color, state, car_type, description, videos))
            if form in ['2','3']:
                for _ in tqdm.tqdm(range(len(self.videos_list2)),desc=f"Loading {self.split} videos"):
                    video = self.videos_list2[_]
                    with open(os.path.join(self.Rope3D_video, video), 'r') as f:
                        anno = json.load(f)[0]
                        car_type = anno['type'].upper()
                        if car_type not in ['CAR','TRUCK','BUS','VAN']:
                            continue
                        # assert car_type != 'VAN', "错误:car_type 不应为 'VAN'"
                        #要加的属性
                        # style_list = list(anno['description'].keys())
                        videoID = 'Rope3D'+anno['videoID']
                        # sequence_id = anno['sequence_id']
                        track_id = anno['track_id']
                        color = anno['color']
                        state = anno['state']
                        description = anno['description']
                        videos = [anno[f"frame{i}"]  for i in range(30)]
                        self.anno_data.append((videoID, track_id, color, state, car_type, description, videos))
        else:
            if form in ['1','3']:
                for _ in tqdm.tqdm(range(len(self.videos_list2)),desc=f"Loading {self.split} videos"):
                    video = self.videos_list2[_]
                    with open(os.path.join(self.Rope3D_video, video), 'r') as f:
                        anno = json.load(f)[0]
                        car_type = anno['type'].upper()
                        if car_type not in ['CAR','TRUCK','BUS','VAN']:
                            continue
                        #要加的属性
                        # style_list = list(anno['description'].keys())
                        videoID = 'Rope3D'+anno['videoID']
                        # sequence_id = anno['sequence_id']
                        track_id = anno['track_id']
                        color = anno['color']
                        state = anno['state']
                        description = anno['description']
                        videos = [anno[f"frame{i}"]  for i in range(30)]
                        self.anno_data.append((videoID, track_id, color, state, car_type, description, videos))
            if form in ['2','3']:
                for _ in tqdm.tqdm(range(len(self.videos_list1)),desc=f"Loading {self.split} videos"):
                    video = self.videos_list1[_]
                    with open(os.path.join(self.Rcooper_video, video), 'r') as f:
                        anno = json.load(f)
                        #要加的属性
                        style_list = list(anno['description'].keys())
                        for i in range(len(style_list)):
                            videoID = 'Rcooper'+anno['videoID']+self.id2style[i+1]
                            # sequence_id = anno['sequence_id']
                            track_id = anno['track_id']
                            color = anno['color']
                            state = anno['state']
                            car_type = anno['type'].upper()
                            description = anno['description'][self.id2style[i+1]]
                            videos = [anno[f"frame{i}"]  for i in range(30)]
                            self.anno_data.append((videoID, track_id, color, state, car_type, description, videos))
 
        # with open(self.anno_path, 'r') as f:
        #     anno = json.load(f)
            
        # for anno_dict in anno:
        #     if anno_dict['im_name'] in self.image_list:  #  anno_dict['im_name'] 是图片名字
        #         instanceID = anno_dict['instanceID']   #对应图片的instanceID
        #         ann_id = anno_dict['ann_id']
        #         text = anno_dict['description']
        #         objectName = anno_dict['objectName']
        #         label_2 = ast.literal_eval(anno_dict["label_2"])# 那一系列的标注
        #         self.anno_data.append((anno_dict['im_name'], instanceID, ann_id, objectName, text, label_2))

        # path configuration
        self.image_dir = self.img_root_dir # 29990  个样本
        # self.calib_dir = os.path.join(self.root_dir, 'calib')   # 我们没有这个 NOTE

        # data augmentation configuration
        self.data_augmentation = True if split in ['train'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)   #这些代码是是用来做数据增强的

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ],
                                       [0,0,0],
                                       [0,0,0],
                                       [0,0,0],
                                       [0,0,0],
                                       [0,0,0],
                                       [0,0,0],])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32 #下采样
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)
    def replace_windows_paths(self,path_list):
        # 判断是否在 Linux 系统上运行
        if os.name == 'posix': 
            path_list[1]=path_list[1].replace('\\', '/') 
            path_list[2]=path_list[2].replace('\\', '/')
            # path_list[3]=path_list[3].replace('\\', '/')               
            # 'posix' 表示类 Unix 系统，Linux 和 macOS 都属于此类
        #     return [path.replace('\\', '/')   for path in path_list if type(path)==type('str') else ]
        # else:
            return path_list

    def get_image(self, path):
        # print(path)
        assert os.path.exists(path)
        return Image.open(path)    # (H, W, 3) RGB mode

    # def get_videos(self,videos):
    #     self.videos_detail = []
    #     for i in range(30):

    #     return self.videos_detail
    def get_label(self, object_2):
        return get_objects_from_label(object_2)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading mono3DVG results and GTs...")
        with open(results_dir, 'r') as f:
            results = json.load(f)

    def __len__(self):
        return self.anno_data.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        videoID, track_id, color, state, car_type, description, videos = self.anno_data[item-1] 
        pattern = pattern = r'^(R\w+?)(?=\d{6})'
        match = re.search(pattern, videoID)
        data_name = match.group(1)
                # language encoding
        text = description
        text = text.lower()
        if self.lstm:
            phrase = self.tokenize_phrase(text)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode language to bert input
            examples = read_examples(text, item)
            features = convert_examples_to_features(examples=examples, seq_length=self.query_len,
                                                    tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            # 1表示真实token,0表示是填充
            word_mask = features[0].input_mask

        targets_text={'word_id': np.array(word_id, dtype=int),
                   'word_mask': np.array(word_mask, dtype=int),
                    'text': text
                    }
        # imgs_jsons_p2s= self.get_videos(videos)
        inputs = []
        targets = []
        all_targets = {}
        all_targets["text"] = targets_text
        P2s = []
        
        for i in range(30):
            single_img_json = videos[i]
            single_img_json = self.replace_windows_paths(single_img_json)
            # 这里是缝补模块
            if data_name == 'Rcooper':
                img_path = os.path.join(self.image_dir[0], os.path.split(single_img_json[2])[1])
                json_path = os.path.join(self.image_dir[0], os.path.split(single_img_json[2])[1])
                calib_path = os.path.join(self.calib_dir[0],os.path.split(single_img_json[1])[1])
                with open(calib_path, 'r') as f:
                    cam_data = json.load(f)

                # 读取内参
                K_list = cam_data['cam_K']
                K = np.array(K_list).reshape(3, 3)  # 转换为 3x3 矩阵
                # 如果想转换为 3x4 投影矩阵 P（假设没有旋转和平移，即相机在世界坐标原点）
                p2_matix = np.hstack((K, np.zeros((3, 1))))
            if data_name == 'Rope3D':
                img_path = os.path.join(self.image_dir[1], os.path.split(single_img_json[2])[1])
                json_path = os.path.join(self.image_dir[1], os.path.split(single_img_json[2])[1])
                calib_path = os.path.join(self.calib_dir[1],os.path.split(single_img_json[1])[1])
                with open(calib_path, 'r') as f:
                    cam_data = json.load(f)

                    # 读取投影矩阵 P
                    P_list = cam_data['P']
                    p2_matix = np.array(P_list).reshape(3, 4)
            ###IMG
            img = self.get_image(img_path)
            img_size = np.array(img.size)
            features_size = self.resolution // self.downsample   
            center = np.array(img_size) / 2
            crop_size, crop_scale = img_size,1
            trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
            # trans,trans_inv = affine_scale_only(crop_size,self.resolution)
            img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1) # NOTE  return1 
            #P2
            old_path = json_path
            new_path = old_path.replace("new1", 
                            "intrinsic_camera_matrix")
            # old_path = img_path
            # old_path = old_path.replace("images\\s110_camera_basler_south1_8mm\\", 
            #                 "labels_point_clouds\\s110_lidar_ouster_north\\intrinsic_camera_matrix\\")
            
            # new_path = old_path.replace(".jpg", 
            #                 ".json")
            # p2 = new_path
            # json_data = open(p2).read()
            # data = json.loads(json_data) 
            # # 所以这里还是使用到相机参数了
            # p2_matix = data["P"]#NOTE  return 2
            #targets
            obj = get_info_from_json(single_img_json,p2_matix)
            # labels encoding
            calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=bool)
            labels = np.zeros((self.max_objs), dtype=np.int8)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
            boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
            
            
            bbox_2d = obj.box2d.copy()
            assert bbox_2d[2] >= bbox_2d[0] and bbox_2d[3] >= bbox_2d[1], f"Invalid bbox: {bbox_2d}"
            # bbox_2d = np.array([single_img_json[6],single_img_json[7],single_img_json[8],single_img_json[9]])  # x min y min x max y max
            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
              # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()               
            center_3d = obj.pos + [0, -obj.h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)  
            # center_3d, _ =obj.rect_to_img(center_3d)  # project 3D center to image plane
            
            center_3d = center_3d[0]  # shape adjustment
            # if (center_3d < 0).any():
            #      raise ValueError(f"center_3d 中的值不能小于 0，当前值为 {center_3d.tolist()}")

            center_3d = affine_transform(center_3d.reshape(-1), trans)

            cls_id = self.cls2id[car_type.upper()]
            labels[0] = cls_id            
            
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[0] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[0] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)

            boxes[0] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[0] = center_3d_norm[0], center_3d_norm[1], l, r, t, b       
            
            depth[0] = obj.pos[-1] * crop_scale
            heading_angle = obj.ry2alpha(obj.ry,(obj.box2d[0] + obj.box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[0], heading_res[0] = angle2class(heading_angle) 
            
              
            src_size_3d[0] = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[car_type]]
            size_3d[0] = src_size_3d[0] - mean_size   
            if obj.truncated_state <= 0.5 and obj.occluded_state <= 2:
                mask_2d[0] = 1
            targets_other =  {
                   'indices': indices,
                   'img_size': img_size,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d}
            
            inputs.append(img)  
            targets.append(targets_other)
            # P2s.append(p2_matix)
        all_targets["other"] = targets
        #all_targets[0]是 text     all_targets[1]是targets  
        if self.split == 'test':
            return inputs, obj.P2, all_targets,videoID[-6:]  

        return inputs, obj.P2, all_targets     
                  
                        

 




def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        # 把原始的自然语言句子分割成模型能够识别的最小词单元序列
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            # 注意有开头和结尾
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        # 转化为ID
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # 生成attention mask
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features
