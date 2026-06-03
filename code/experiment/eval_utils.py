import json
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union > 0 else 0.0


def overlap_ratio(det, anchor):
    ax1,ay1,ax2,ay2 = det; bx1,by1,bx2,by2 = anchor
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    area  = max(1,(ax2-ax1)*(ay2-ay1))
    return inter/area


def expand_bbox(bbox, margin, w, h):
    x1,y1,x2,y2 = bbox
    dx,dy = int((x2-x1)*margin), int((y2-y1)*margin)
    return (max(0,x1-dx),max(0,y1-dy),min(w,x2+dx),min(h,y2+dy))


def enclosing(boxes):
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes))


def prec_rec_f1(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return round(p,4), round(r,4), round(f,4)


CAT_MAP = {1:"coverall",2:"face_shield",3:"gloves",4:"goggles",5:"mask"}
CLUSTER_EXPAND   = 0.10
PERSON_IOU_THRESH = 0.50
PPE_MATCH_IOU    = 0.50
PPE_ASSIGN_OVR   = 0.20


def cluster_gt_persons(anns, img_w, img_h):
    boxes = []
    for a in anns:
        x,y,bw,bh = a["bbox"]
        boxes.append((x, y, x+bw, y+bh))
    n = len(boxes)
    parent = list(range(n))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(i)]=find(j)
    expanded = [expand_bbox(b, CLUSTER_EXPAND, img_w, img_h) for b in boxes]
    for i in range(n):
        for j in range(i+1,n):
            if iou(expanded[i], expanded[j])>0: union(i,j)
    clusters = defaultdict(list)
    for i in range(n): clusters[find(i)].append(i)
    persons = []
    for idxs in clusters.values():
        ann_group = [anns[i] for i in idxs]
        bbox = enclosing([boxes[i] for i in idxs])
        persons.append({"bbox": bbox, "anns": ann_group})
    return persons


def gt_worn_per_person(gt_persons, img_w, img_h):
    result = []
    for gp in gt_persons:
        worn = set()
        for a in gp["anns"]:
            worn.add(CAT_MAP.get(a["category_id"], str(a["category_id"])))
        result.append({"bbox": gp["bbox"], "worn": worn})
    return result


def yolo_bbox(parts, img_w, img_h):
    _, xc, yc, w, h = map(float, parts)
    x1 = int((xc-w/2)*img_w); y1 = int((yc-h/2)*img_h)
    x2 = int((xc+w/2)*img_w); y2 = int((yc+h/2)*img_h)
    return (x1,y1,x2,y2)


def load_yolo_person_bboxes(label_path, img_w, img_h, person_class=0):
    if not Path(label_path).exists(): return []
    bboxes = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0])==person_class:
            bboxes.append(yolo_bbox(parts, img_w, img_h))
    return bboxes


def load_cppe5_gt(ann_path):
    coco = json.load(open(ann_path))
    id_to_img  = {i["id"]: i for i in coco["images"]}
    id_to_anns = defaultdict(list)
    for a in coco["annotations"]:
        id_to_anns[a["image_id"]].append(a)
    return coco["images"], id_to_img, id_to_anns
