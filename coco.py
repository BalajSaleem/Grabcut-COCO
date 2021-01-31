from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from grabcut import GC

dataDir='./COCOdataset2017'
dataType='val'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
iou_scores = []
bbox_increase = 0.1
trial_batch_size = 50

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

# Initialize the COCO api for instance annotations
coco=COCO(annFile)


# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

#print(cats)

selected_cats = ['person', 'car', 'horse', 'sheep', 'stop sign', 'airplane' ]
result_ious = {}
# Define the classes (out of the 81) which you want to see. Others will not be shown.

for cat in selected_cats:
    counter = 0
    iou_scores = []
    print("RUNNING TRIAL FOR CATEGORY: " + cat)
    for i in range(trial_batch_size):
        
        #cat_item = random.choice(cats[:10])['name']
        filterClasses = [cat]

        # Fetch class IDs only corresponding to the filterClasses
        catIds = coco.getCatIds(catNms=cat) 
        #print("category: " + str(cat_item))
        # Get all images containing the above Category IDs
        imgIds = coco.getImgIds(catIds=catIds)
        #print("Number of images containing all the  classes:", len(imgIds))

        # load and display a random image
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        I = cv2.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        #coco.showAnns(anns)

        [x,y,w,h] = anns[0]['bbox']
        I_cpy = I
        mask = np.zeros((img['height'],img['width']), dtype=np.uint8)
        # extract only one mask
        mask = np.maximum(coco.annToMask(anns[0]), mask)
        segmented_img = cv2.bitwise_and(I, I, mask = mask)
        
        #run GC
        gc = GC()
        gc.loadImage(inputImage = I)
        gc.reset()
        new_x = (x-(bbox_increase)*w) if (x-(bbox_increase)*w) >= 0 else 1
        new_y = (y-(bbox_increase)*h) if (y-(bbox_increase)*h) >= 0 else 1

        gc_output = gc.segmentWithBbox( inRect=tuple([int(new_x),int(new_y),int(w + (bbox_increase)*w),int(h+(bbox_increase)*h)]))#bounding box increase = 10%
        
        intersection = np.logical_and(segmented_img, gc_output)
        union = np.logical_or(segmented_img, gc_output)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)
        #print("IOU SCORE: " + str(iou_score))

        if((i%5) == 0):
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            fig.suptitle("GC ON COCO, IOU SCORE: " + str(iou_score) )
            new_x = (x-(bbox_increase)*w) if (x-(bbox_increase)*w) >= 0 else 1
            new_y = (y-(bbox_increase)*h) if (y-(bbox_increase)*h) >= 0 else 1
            cv2.rectangle(I_cpy, (int(x), int(y)), ((int(x+w), int(y+h))), (0,255,0), 4)
            cv2.rectangle(I_cpy, (int(new_x), int(new_y)), ((int(x+ (w + bbox_increase*w)), int(y+ (h + bbox_increase*h)))), (255,0,0), 4)
            ax1.imshow(cv2.cvtColor(I_cpy, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
            ax3.imshow(cv2.cvtColor(gc_output, cv2.COLOR_BGR2RGB))
            plt.savefig(cat + str(counter))
            counter += 1
            plt.clf()

    
    iou_scores = np.array(iou_scores)
    iou_score = np.mean(iou_scores)
    result_ious.update({cat: str(iou_score)})
    print( cat +  " -- " + str(iou_score))

print("----FINAL IOUS----")
print(result_ious)
