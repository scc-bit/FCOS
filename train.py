from model.fcos import FCOSDetector
import torch
from dataloader.VOC_dataset import VOCDataset
import math,time
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_dataset=VOCDataset(r"FCOS/VOCdevkit/VOC2012",resize_size=[800,1024],split='train')

model=FCOSDetector(mode="training").cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

BATCH_SIZE=12
EPOCHS=30
WARMPUP_STEPS_RATIO=0.12
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn)

steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMPUP_STEPS=TOTAL_STEPS*WARMPUP_STEPS_RATIO

GLOBAL_STEPS=1
LR_INIT=5e-5
LR_END=1e-6

trainwriter=SummaryWriter(log_dir="FCOS/logs")

def lr_func():
    if GLOBAL_STEPS<WARMPUP_STEPS:
        lr=GLOBAL_STEPS/WARMPUP_STEPS*LR_INIT
    else:
        lr=LR_END+0.5*(LR_INIT-LR_END)*(
            (1+math.cos((GLOBAL_STEPS-WARMPUP_STEPS)/(TOTAL_STEPS-WARMPUP_STEPS)*math.pi))
        )
    return float(lr)

model.train()

for epoch in range(EPOCHS):
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
        
        trainwriter.add_scalar("train_loss",loss.item(),global_step=GLOBAL_STEPS)
        trainwriter.add_scalar("lr",lr,global_step=GLOBAL_STEPS)

        GLOBAL_STEPS+=1
    
    torch.save(model.state_dict(),"FCOS/weights/FCOS_VOC2012_epoch%d.pth"%(epoch+1))
    






