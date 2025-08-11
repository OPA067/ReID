import cv2
import cvzone
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
import os
from PIL import Image
import torchvision.transforms as T
import os.path as op
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
import matplotlib.pyplot as plt
from utils.iotools import load_train_configs
import warnings
warnings.filterwarnings("ignore")

# load yolo coco class_list
with open("../coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
cv2.namedWindow('Online')
cv2.setMouseCallback('Online', RGB)

def online(detect_model, retrieval_model, tar_img):

    device = next(retrieval_model.parameters()).device
    tar_img = tar_img.to(device)
    tar_img = tar_img.unsqueeze(0)

    retrieval_model.eval()
    with torch.no_grad():
        tar_feat = retrieval_model.encode_image(tar_img)
        tar_feat = tar_feat / tar_feat.norm(dim=-1, keepdim=True)

    all_video_path = '{candidate videos path}'
    video_name_list = os.listdir(all_video_path)

    frame_count = 0
    best_sims_list = [{'sim': -1.0, 'image': None} for _ in range(1, 11)]
    for video_name in tqdm(video_name_list):
        video_path = all_video_path + video_name
        cap = cv2.VideoCapture(video_path)
        video_max_sims = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1500, 768)) # maybe #
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            results = detect_model.predict(frame)
            detections = results[0].boxes.data
            px = pd.DataFrame(detections.cpu().numpy()).astype("float")

            pos_imgs, can_imgs = [], []
            for index, row in px.iterrows():
                x1, y1 = int(row[0]), int(row[1])
                x2, y2 = int(row[2]), int(row[3])
                confidence = row[4]
                c = class_list[int(row[5])]
                # if c in ['person'] and confidence > 0.5: # choose class
                if confidence > 0.5:
                    pos_imgs.append([x1, y1, x2, y2, c])
                    can_img = frame[y1:y2, x1:x2]
                    can_img = transform(Image.fromarray(can_img))
                    can_imgs.append(can_img)
            if len(can_imgs) == 0:
                cv2.imshow("Online", frame)
            else:
                can_imgs = torch.stack(can_imgs).to(device)
                with torch.no_grad():
                    can_feat = retrieval_model.encode_image(can_imgs)
                    can_feat = can_feat / can_feat.norm(dim=-1, keepdim=True)
                sims = tar_feat @ can_feat.T
                sims = sims.squeeze(0).cpu().detach().numpy()  # [1, B] of GPU => [B] of CPU
                video_max_sims.append(np.max(sims))
                for i, (pos, sim) in enumerate(zip(pos_imgs, sims)):
                    for j in range(len(best_sims_list)):
                        if sim < best_sims_list[-1]['sim']:
                            break
                        if sim > best_sims_list[j]['sim']:
                            best_sims_list.insert(j, {'sim': sim, 'image': frame[pos[1]:pos[3], pos[0]:pos[2]]})
                            break

                    if len(best_sims_list) > 10:
                        best_sims_list.pop()

                    cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), (255, 0, 0), 1)
                    cvzone.putTextRect(frame, f'{pos[4]}_{sim:.4f}', (pos[0], pos[1]), 0.8, 1)

                cv2.imshow("Online", frame)

            # Next one (video)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # save object frame location
        plt.rcParams['font.size'] = 10
        x = np.arange(len(video_max_sims))
        plt.figure(figsize=(10, 5))
        plt.plot(x, video_max_sims, marker='o', linestyle='-', color='b', alpha=0.5)
        plt.xlabel('Frame * 3')
        plt.ylabel('Max Similarity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./{"save path"}/{video_name.split(".")[0]}_max_sims_line_chart.png')

        # save top10 object images
        for idx, item in enumerate(best_sims_list):
            sims, image = item['sim'], item['image']
            cv2.imwrite(f'./{"save path"}/img_{idx}.jpg', image)

        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID")
    sub = '{Model save dir instead of .pth}'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('ReID', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir = sub

    # 1. load retrieval model
    logger.info("===>>> Load retrieval model....")
    retrieval_model = build_model(args)
    checkpointer = Checkpointer(retrieval_model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    retrieval_model = retrieval_model.cuda()

    # 2. load detect model
    logger.info("===>>> Load detect model....")
    detect_model = YOLO('pretrain_models/yolov8l.pt')

    # 3. load target image
    logger.info("===>>> Load target image....")
    tar_img = Image.open("target_image.jpg").convert('RGB')
    tar_img = transform(tar_img)

    # 4. begin online reid...
    online(detect_model, retrieval_model, tar_img)