import torch
import numpy as np
import cv2
from model.config import InferenceConfig
from model.metric  import sort_by_score,eval_ap_2d
from model.fcos import FCOSDetector
from dataloader.VOC_dataset import VOCDataset
    
if __name__=="__main__":
    
    
    BATCH_SIZE=1
    eval_dataset=VOCDataset("FCOS/VOCdevkit/VOC2012",resize_size=[800,1024],split='val2')
    print("eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=eval_dataset.collate_fn)

    model=FCOSDetector(mode="inference",config=InferenceConfig)
    model.load_state_dict(torch.load("/root/FCOS/weights/mini_voc2012_epoch3.pth",map_location=torch.device('cuda:0')))
    model.cuda().eval()
    print("successfully loading model")

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0
    for img,boxes,classes in eval_loader:
        with torch.no_grad():
            out=model(img.cuda())
            pred_scores.append(out[0][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_boxes.append(out[2][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        print(num,end='\r')


    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME))
    print("all classes AP:\n",all_AP)
    for key,value in all_AP.items():
        print('ap of {0}:{1:.5f}'.format(eval_dataset.id2name[int(key)],value))
    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(eval_dataset.CLASSES_NAME)-1)
    print("mAP:%.5f\n"%mAP)

