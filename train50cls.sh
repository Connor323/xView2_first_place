echo "training seresnext50 classification model with seeds 0-2"
python train50_cls_cce.py 0 0
python train50_cls_cce.py 1 0
python train50_cls_cce.py 2 0
python tune50_cls_cce.py 0 0
python tune50_cls_cce.py 1 0
python tune50_cls_cce.py 2 0
echo "Model seresnext50 classification trained!"