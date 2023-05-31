import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataloader.VOC_dataset import VOCDataset
import time
from model.config import InferenceConfig

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    

if __name__=="__main__":
        
    model=FCOSDetector(mode="inference",config=InferenceConfig)
    model.load_state_dict(torch.load("/root/FCOS/weights/voc2012_20head_epoch1_loss1.5069.pth",map_location=torch.device('cpu')))
    model.cuda().eval()
    print("loading model successfully.")

    import os
    root="FCOS/test_images/input/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[800,1024])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)

        img_t=transforms.ToTensor()(img)

        img1= transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225],inplace=True)(img_t)
 
        img1=img1.cuda()

        start_t=time.time()
        with torch.no_grad():

            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("successfuly processing img, cost time %.2f ms"%cost_t)

        scores,classes,boxes=out


        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            img_pad=cv2.putText(img_pad,"%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
        cv2.imwrite("FCOS/test_images/output/"+name,img_pad)





