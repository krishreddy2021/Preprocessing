import torch
import torchvision
import numpy as np

def main(ground_truth, localizer):

    print("Main Script")

    # Preprocess arrays from dictionary provided 

    sorted_gt = dict(sorted(ground_truth.items()))
    sorted_localizer = dict(sorted(localizer.items()))

    # Convert sorted dictionaries to arrays 

    gt_arr = []
    localizer_arr = []

    for key in sorted_gt:
        gt_arr.append(sorted_gt[key])
        localizer_arr.append(sorted_localizer[key])

    mod_groundT = modify_bound_coors(gt_arr)
    mod_images = modify_bound_coors(localizer_arr)

    #iou_matrixes = [iOU(mod_images[i], ground_truth[i]) for i in range(len(mod_images))]

    iou_matrixes = np.array([])

    for i in range(len(mod_images)):
        iou_matrixes = np.append(iou_matrixes, iOU(mod_images[i], mod_groundT[i]))
    

def modify_bound_coors(input):

    mod_images = []

    for i in range(len(input)):

        bounding_boxes = []
        for j in range(len(input[i])):

            if(len(input[i][j])>0):
            
                #(input[i][j][0], input[i][j][1]) is top left and (input[i+input[2]][j][0], input[i][j+input[3]][1]) is bottom right

                bounding_boxes.append([input[i][j][0], input[i][j][1], input[i][j][0] + input[i][j][2], input[i][j][1] + input[i][j][3]])

        mod_images.append(bounding_boxes)

    return mod_images
    
def iOU(localizer, ground_truth):

     # Define the two arrays of bounding boxes (in the format [x1, y1, x2, y2])

    image1 = torch.tensor(localizer)
    image2 = torch.tensor(ground_truth)

    # Calculate the IoU between all pairs of boxes using the bbox_overlaps function
    iou_matrix = torchvision.ops.box_iou(image1, image2)

    # Print the result
    #print(iou_matrix)

    return iou_matrix


if __name__ == "__main__":

    groundTruth = dict()
    localizer = dict()

    # Test cases 
    groundTruth["1"] = [[1,2,3,4], []]
    groundTruth["2"] = [[2,2,4,5]]
    localizer["1"] = [[1,1,2,2],[2,1,5,4]]
    localizer["2"] = [[2,4,3,5]]

    main(groundTruth, localizer)