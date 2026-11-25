
import os, sys, csv
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))      
from fusion_metrics import entropy, std_dev, spatial_frequency, mi_fusion

ir_dir    = 'Dataset/testsets/MSRS/IR'
vi_dir    = 'Dataset/testsets/MSRS/VI_Y'
fused_dir = 'results/SwinFusion_MSRS'

names = sorted(os.listdir(fused_dir))
rows = []
for name in names:
    ir = cv2.imread(os.path.join(ir_dir, name), 0)
    vi = cv2.imread(os.path.join(vi_dir, name), 0)
    f  = cv2.imread(os.path.join(fused_dir, name), 0)
    assert ir.shape == vi.shape == f.shape, name   

    rows.append({
        'name': name,
        'EN':  entropy(f),
        'SD': std_dev(f),
        'SF': spatial_frequency(f),
        'MI': mi_fusion(ir,vi,f)
    })

EN_mean = np.mean([r['EN'] for r in rows])
SD_mean = np.mean([r['SD'] for r in rows])
SF_mean = np.mean([r['SF'] for r in rows])
MI_mean = np.mean([r['MI'] for r in rows])

for m in ['EN','SD','SF','MI']:
    print(f"{m} : {np.mean([r[m] for r in rows]):.4f}")


with open('results/metrics_SwinFusion_MSRS.csv', 'w', newline='') as fin:
    writer = csv.DictWriter(fin, fieldnames=['name', 'EN', 'SD', 'SF', 'MI'])
    writer.writeheader()      # первая строка: name,EN,SD,SF,MI
    writer.writerows(rows)    # 361 строка, по одной на словарь


# TODO: напечатать среднее каждой метрики по всем картинкам
# TODO (по желанию): сохранить rows в results/metrics_SwinFusion_MSRS.csv через csv.DictWriter