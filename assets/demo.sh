
# Detection (OD, IS)
python3 projects/UNINEXT/demo.py --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--input assets/demo.jpg --output demo/detection.jpg \
--task detection \
--opts MODEL.WEIGHTS outputs/image_joint_r50/model_final.pth

# Grounding (REC, RES)
python3 projects/UNINEXT/demo.py --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--input assets/demo.jpg --output demo/left.jpg \
--task grounding --expressions "person on the left" \
--opts MODEL.WEIGHTS outputs/image_joint_r50/model_final.pth

python3 projects/UNINEXT/demo.py --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--input assets/demo.jpg --output demo/middle.jpg \
--task grounding --expressions "middle person" \
--opts MODEL.WEIGHTS outputs/image_joint_r50/model_final.pth

python3 projects/UNINEXT/demo.py --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--input assets/demo.jpg --output demo/white.jpg \
--task grounding --expressions "person in white" \
--opts MODEL.WEIGHTS outputs/image_joint_r50/model_final.pth

python3 projects/UNINEXT/demo.py --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--input assets/demo.jpg --output demo/bat.jpg \
--task grounding --expressions "baseball bat on the right" \
--opts MODEL.WEIGHTS outputs/image_joint_r50/model_final.pth