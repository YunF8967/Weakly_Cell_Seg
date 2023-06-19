import os
import cv2
import tifffile as tiff
import random


def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # tiff.imshow(img)

def read_png(path):
    img = cv2.imread(path)
    return img

def read_tiff(path):
    img = tiff.imread(path) # np.ndarray
    return img

def generatePatch(root, tag): #eg, root: data/.../tnbc

    for curDir, _, files in os.walk(root):

        if 'img' in curDir: # ../tnbc/img_test, ../tnbc/img_train

            for file in files:

                img_path = curDir + '/'+file   # data/cell-seg-data/tnbc/img_test/xxx.png
                ann_path = img_path.replace('img', 'ann')
                ann_path = ann_path.replace('.png', '.tif') # data/cell-seg-data/tnbc/ann_test/xxx.tif

                img = read_png(img_path)
                ann = read_tiff(ann_path)
                
                '''assert initial image size'''
                if 'fluo' in curDir:
                    assert img.shape[0] == 520
                    assert img.shape[1] == 696
                    assert ann.shape[0] == 520
                    assert ann.shape[1] == 696
                elif 'tnbc' in curDir: 
                    assert img.shape[0] == 512
                    assert img.shape[1] == 512
                    assert ann.shape[0] == 512
                    assert ann.shape[1] == 512
                else:
                    print("Error: <{}> Not Fluo or TNBC".format(file))
                    break

                '''cut & save'''
                # create patch dirs
                patchImgDir = curDir.replace(tag, tag+"_patch")
                if not os.path.exists(patchImgDir):
                    os.makedirs(patchImgDir)

                patchAnnDir = patchImgDir.replace("img", "ann")
                if not os.path.exists(patchAnnDir):
                    os.makedirs(patchAnnDir)

                # for each img, generate (within-range) random number of patches
                len0 = 256  
                len1 = 256  # the 2 side length of each patches
                for i in range(0, random.randint(10,15)):
                    # position of top left corner of the patch
                    a0 = random.randint(0, img.shape[0]-len0)
                    a1 = random.randint(0, img.shape[1]-len1)
                    patchImg = img[a0:a0+len0, a1:a1+len1]
                    patchAnn = ann[a0:a0+len0, a1:a1+len1]

                    patchBaseName = os.path.splitext(file)[0]+'_'+str(i)

                    patchImgPath = patchImgDir + '/' + patchBaseName + '.png'
                    patchAnnPath = patchAnnDir + '/' + patchBaseName + '.tif'

                    cv2.imwrite(patchImgPath, patchImg)
                    tiff.imwrite(patchAnnPath, patchAnn)
            
    


# '''tnbc'''
# root = "data/cell-seg-data/tnbc/"

# '''fluo'''
# root = "data/cell-seg-data/fluo/"

# main("data/cell-seg-data/fluo/")


# img= read_tiff("data/cell-seg-data/test_tnbc/ann/Slide_01_1.tif")
# patches = cut(img, 2, 2)
# savePatches("data/cell-seg-data/patch_test_tnbc/ann/", 'Slide_01_1', patches)
# check = read_png("data/cell-seg-data/patch_test_tnbc/ann/Slide_01_1_1.tif")
# inp = read_png("data/cell-seg-data/test_tnbc/ann/Slide_01_1.tif")





