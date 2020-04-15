echo "training dpn92 classification model with seeds 0-2"
python train92_cls_cce.py 0 0
python train92_cls_cce.py 1 0
python train92_cls_cce.py 2 0
python tune92_cls_cce.py 0 0
python tune92_cls_cce.py 1 0
python tune92_cls_cce.py 2 0
echo "Model dpn92 classification trained!"