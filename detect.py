import torch
import cv2
import numpy as np

from models.YOLOV7 import YOLOV7
from utils.utils import non_max_suppression, letterbox, scale_coords, load_pretrain_model


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def postprocess(outputs, B, num_classes, anchors, masks, strides=(8, 16, 32)):
    anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32))
    masks = torch.from_numpy(np.asarray(masks, dtype=np.int))
    num_output = num_classes + 5

    z = []
    for i, x in enumerate(outputs):
        bs, _, ny, nx =  x.shape
        x = x.view(bs, B, num_output, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        grid = make_grid(nx, ny)
        y = x.sigmoid()
        anchor_grid = anchors[masks[i]].view(1, -1, 1, 1, 2)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * strides[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
        z.append(y.view(bs, -1, num_output))

    return torch.cat(z, 1)


def preprocess(img_path, net_size):
    img0 = cv2.imread(img_path)  # BGR

    # Letterbox
    img = letterbox(img0, net_size)[0]

    # BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # norm to torch
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, img0


def model_init(model_path, B=3, C=80):
    # load moel
    model = YOLOV7(B=B, C=C)
    load_pretrain_model(model, model_path)
    model.eval()
    return model


if __name__ == '__main__':
    # load moel
    checkpoint_path = 'weights/yolov7_samylee.pth'
    B, C = 3, 80
    model = model_init(checkpoint_path, B, C)

    # params init
    net_size = 640
    anchors = [[12, 16], [19, 36], [40, 28],
               [36, 75], [76, 55], [72, 146],
               [142, 110], [192, 243], [459, 401]]
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = (8, 16, 32)
    conf_thresh = 0.25
    iou_thresh = 0.45

    # coco
    with open('assets/coco.names', 'r') as f:
        classes = [x.strip().split()[0] for x in f.readlines()]

    # preprocess
    img_path = 'demo/bus.jpg'
    img, im0 = preprocess(img_path, net_size)

    # forward
    outputs = model(img)

    # postprocess
    pred = postprocess(outputs, B, C, anchors, masks, strides)
    pred = non_max_suppression(pred, conf_thresh, iou_thresh)

    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{classes[int(cls)]} {conf:.2f}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
                cv2.putText(im0, label, c1, 0, 0.6, (0,255,255), 2)
    # cv2.imwrite('assets/result1.jpg', im0)
    cv2.imshow('test', im0)
    cv2.waitKey(0)
