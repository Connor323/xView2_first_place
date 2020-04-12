echo "training dpn92 classification model with seeds 0-2"
python train92_cls_cce.py 0 1
python train92_cls_cce.py 1 1
python train92_cls_cce.py 2 1
python tune92_cls_cce.py 0 1
python tune92_cls_cce.py 1 1
python tune92_cls_cce.py 2 1
echo "Model dpn92 classification trained!"