from model.fcos import FCOSDetector
import torch
from dataloader.VOC_dataset import VOCDataset
from model.metric  import sort_by_score,eval_ap_2d
import math,time
from torch.utils.tensorboard import SummaryWriter
from model.config import InferenceConfig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_dataset=VOCDataset(r"FCOS/VOCdevkit/VOC2012",resize_size=[800,1024],split='train')
val_dataset=VOCDataset(r"FCOS/VOCdevkit/VOC2012",resize_size=[800,1024],split='val')

model=FCOSDetector(mode="training").cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

model_val=FCOSDetector(mode="inference", config=InferenceConfig) 

BATCH_SIZE=12
EPOCHS=30  
WARMPUP_STEPS_RATIO=0.12
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=True,collate_fn=val_dataset.collate_fn)
steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMPUP_STEPS=TOTAL_STEPS*WARMPUP_STEPS_RATIO


GLOBAL_STEPS=1
LR_INIT=5e-5
LR_END=1e-6

writer=SummaryWriter(log_dir="FCOS/logs")

def lr_func():
    if GLOBAL_STEPS<WARMPUP_STEPS:
        lr=GLOBAL_STEPS/WARMPUP_STEPS*LR_INIT
    else:
        lr=LR_END+0.5*(LR_INIT-LR_END)*(
            (1+math.cos((GLOBAL_STEPS-WARMPUP_STEPS)/(TOTAL_STEPS-WARMPUP_STEPS)*math.pi))
        )
    return float(lr)
       

def evaluate(model, data_loader):
    print("evaluating...epoch:%d"%(epoch+1))
    print("successfully loading model")
    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0
    for img,boxes,classes in data_loader:
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
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(val_dataset.CLASSES_NAME))
    print("all classes AP:\n",all_AP)
    for key,value in all_AP.items():
        print('ap of {0} :{1:.5f}'.format(val_dataset.id2name[int(key)],value))
    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(val_dataset.CLASSES_NAME)-1)
    print("mAP:%.5f\n"%mAP)
    return mAP
    

if __name__=="__main__":
    
    model.train()
    for epoch in range(EPOCHS):
        print("training...epoch:%d"%(epoch+1))
        for epoch_step,data in enumerate(train_loader):
            batch_imgs,batch_boxes,batch_classes=data
            batch_imgs=batch_imgs.cuda()
            batch_boxes=batch_boxes.cuda()
            batch_classes=batch_classes.cuda()

            lr=lr_func()
            for param in optimizer.param_groups:
                param['lr']=lr
        
            start_time=time.time()

            optimizer.zero_grad()
            losses=model([batch_imgs,batch_boxes,batch_classes])
            loss=losses[-1]
            loss.backward()
            optimizer.step()

            end_time=time.time()
            cost_time=int((end_time-start_time)*1000)

            print("global_steps:%d epoch:%d steps:%d/%d train_loss:%.4f cost_time:%dms lr=%.4e"%\
                (GLOBAL_STEPS,epoch+1,epoch_step,steps_per_epoch,loss.item(),cost_time,lr))
            
            writer.add_scalar("train_loss",loss.item(),GLOBAL_STEPS)
            GLOBAL_STEPS+=1 

        torch.save(model.state_dict(),"FCOS/weights/FCOS_VOC2012_epoch%d.pth"%(epoch+1))
        model_val.load_state_dict(torch.load("FCOS/weights/FCOS_VOC2012_epoch%d.pth"%(epoch+1)))
        model_val.cuda()
        with torch.no_grad():
            mAP = evaluate(model_val, val_loader)
            writer.add_scalar("mAP", mAP,epoch+1) 
            test_loss = 0
            for i, data in enumerate(val_loader):
                batch_imgs, batch_boxes, batch_classes = data
                batch_imgs = batch_imgs.cuda()
                batch_boxes = batch_boxes.cuda()
                batch_classes = batch_classes.cuda()
                        
                loss = model([batch_imgs, batch_boxes, batch_classes])[-1]
                test_loss += loss
            test_loss /= len(val_dataset)
            writer.add_scalar("val_loss", test_loss.item(),epoch+1)
            torch.cuda.empty_cache()
            
            
    
    






