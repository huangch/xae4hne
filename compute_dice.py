from os import listdir
from os.path import isfile, join
from numpy import zeros, uint8, sum
import cv2
from shutil import copyfile
from scipy.ndimage.morphology import binary_fill_holes
# 
# input_path = 'dataset/manual_seg/boundary'
# boundarymask_path = 'dataset/manual_seg/boundary_mask'
# output_path = 'dataset/manual_seg/boundary_center'
# image_path = 'dataset/normalized_data'
# data_path = 'dataset/seg_data'
# 
# file_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
# 
# for f in file_list:
#     img_id = int(f[5:][::-1][6:][::-1])
#     loc_id = int(f[::-1][4])
#     img = cv2.imread(join(input_path, f))[..., 1]
#     
#     seg_gt = 255*binary_fill_holes(img/255).astype(int)
#     
#     cv2.imwrite(join(boundarymask_path, f), seg_gt)
#     
#     M = cv2.moments(img)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     result = zeros(img.shape, dtype=uint8)
#     result[cY, cX] = 255
#     cv2.imwrite(join(output_path, f), result)
#     copyfile(join(image_path, 'im'+str(img_id).zfill(3)+'.tif'), join(data_path, f))
#     
#     
    
gt_path = 'dataset/manual_seg/boundary_mask'
seg_path = 'segmentation_masks'
k = 1

file_list = [f for f in listdir(gt_path) if isfile(join(gt_path, f))]

total_dice = 0
total_cnt = 0
for f in file_list:
    gt = cv2.imread(join(gt_path, f))[...,0]/255
    
    M = cv2.moments(gt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    if cx < 11 or cx > 88 or cy < 8 or cy > 88:
        continue
    
    seg = cv2.imread(join(seg_path, f+'f'))[...,0]/255
    
    dice = sum(seg[gt==k]==k)*2.0 / (sum(seg[seg==k]==k) + sum(gt[gt==k]==k))

    print '{}, {}, {}: Dice similarity score is {}'.format(f, cx, cy, dice)

    total_dice += dice
    total_cnt += 1
    
print(total_dice/total_cnt)
    