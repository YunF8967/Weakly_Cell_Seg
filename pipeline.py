from pycococreatortools import pycococreatortools
import tifffile as tiff
import cv2
import os
import numpy as np
import json
import datetime

from patcher import generatePatch

'''
This script: Original img -> Patches -> Generate COCO json
'''


def generateCOCO(root, tag): #, IMAGE_DIR, ANNOTATION_DIR, output_path):
    INFO = {
        "description": "Cell-Instance-Seg",
        "url": "",
        "version": "0.1.0",
        "year": 2023,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 0,
            'name': 'background',
            'supercategory': 'background',
        },
        {
            'id': 1,
            'name': 'cell',
            'supercategory': 'cell',
        }
    ]

    coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []  # “一定要有cat=0的分类！但如果设bg=0，会使mAP值变小=> 改进：不设bg=0但把cell标签-1”
        }
    
    for curDir, _, files in os.walk(root):   
        
        if 'img' in curDir:
            image_id = 1
            segmentation_id = 1
            coco_output["images"] = []
            coco_output["annotations"] = []
            # print("curDir: ", curDir)
            
            for file in files: #file: xxx.png
                # basename_no_extension = os.path.splitext(file)[0]

                img_path = curDir + '/'+file   # data/cell-seg-data/../img_test/xxx.png
                ann_path = img_path.replace('img', 'ann')
                ann_path = ann_path.replace('.png', '.tif') # data/cell-seg-data/../ann_test/xxx.tif
                # print("img path: ", img_path)
                # print("ann path: ", ann_path)

                # annotations[] - (1)
                ann = tiff.imread(ann_path)

                values = np.unique(ann) # cell vals
                values = np.delete(values, 0) # remove value of bg pixels
                # print("values: ", values)

                if len(values) == 0:
                    continue

                # images[]
                img = cv2.imread(img_path)
                img_size = [img.shape[1], img.shape[0]]
                    #会影响pycococreatortools里的resize process；
                    # 也会影响self_segentor里的 training process

                image_info = pycococreatortools.create_image_info(
                    image_id, file, img_size)
                coco_output["images"].append(image_info)

                # annotations[] - (2)
                for cell_id in values:
                    category_info = {'id': 1, 'is_crowd': 0}

                    ann_bimask = (ann == cell_id) * ann
                    ann_bimask[ann_bimask != 0] = 1
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, ann_bimask, 
                        img_size, tolerance=2)  
                    
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

                image_id = image_id + 1


            fileName = curDir[curDir.rfind('\\')+1:].replace('img', tag) # eg, img_test => tnbc_test
            jsonPath = root + '/' + fileName + '.json'
            # print("fileName: ", fileName)
            # print("jsonPath: ", jsonPath)
            with open(jsonPath, 'w') as output_json_file:
                json.dump(coco_output, output_json_file)



def main():
    '''fluo'''
    root = "data/cell-seg-data/fluo"
    generatePatch(root, "fluo")
    generateCOCO(root+"_patch", "fluo")

    '''tnbc'''
    # root = "data/cell-seg-data/tnbc"
    # generatePatch(root, "tnbc")
    # generateCOCO(root+"_patch", "tnbc")

if __name__ == '__main__':
    main()
