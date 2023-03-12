
export PYTHONPATH=$PYTHONPATH:`pwd`

python3 -m bdd100k.eval.run -t seg_track -g datasets/bdd/labels/seg_track_20/bitmasks/val -r seg_track