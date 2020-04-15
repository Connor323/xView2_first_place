echo "training senet154 classification model with seeds 0-2"
python train154_cls_cce.py 0 0
python train154_cls_cce.py 1 0
python train154_cls_cce.py 2 0
python tune154_cls_cce.py 0 0
python tune154_cls_cce.py 1 0
python tune154_cls_cce.py 2 0
echo "Model senet154 classification trained!"