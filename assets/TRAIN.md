# Tutorial for Training

Every experiment is defined using a yaml file under the [projects/UNINEXT/configs](../projects/UNINEXT/configs) folder. UNINEXT has three training stages: (1) Object365 pretraining (2) image-level joint training (3) video-level joint training. Corresponding yaml files start with `obj365v2_32g`, `image_joint`, and `video_joint_r50` respectively. By default, we train UNINEXT using 32 or 16 A100 GPUs. Besides, if users are only interested in part of tasks like object detection (OD) and instance segmentation (IS), we also provide yaml files of single tasks, which start with `single_task`. By default, we run these experiments on a single node of 8 GPUs. 

## Single-Node Training

On a single node with 8 GPUs, run 
```
python3 launch.py --nn 1 --uni 1 \
--config-file projects/UNINEXT/configs/${EXP_NAME}.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME} \
```
${EXP_NAME} should be replaced with a specific name. It's worth noting that video-level tasks depends on the weights of image-level tasks.

<table>
  <tr>
    <th>Task</th>
    <th>YAML</th>
    <th>Property</th>

  </tr>
  <tr>
    <td>OD&IS</td>
    <td>single_task_det</td>
    <td>image</td>
  </tr>
  <tr>
    <td>REC&RES</td>
    <td>single_task_rec</td>
    <td>image</td>
  </tr>
  <tr>
    <td>SOT&VOS</td>
    <td>single_task_sot</td>
    <td>video</td>
  </tr>
  <tr>
    <td>VIS</td>
    <td>single_task_vis</td>
    <td>video</td>
  </tr>
  <tr>
    <td>RVOS</td>
    <td>single_task_rvos</td>
    <td>video</td>
  </tr>

</table>


## Multiple-Node Training
Take image-level joint training of UNINEXT with ResNet-50 backbone as an example, run the following commands.

On node 0, run
```
python3 launch.py --nn 2 --port <PORT> --worker_rank 0 --master_address <MASTER_ADDRESS> \
--uni 1 --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--resume OUTPUT_DIR ./image_joint_r50
```
On node 1, run
```
python3 launch.py --nn 2 --port <PORT> --worker_rank 1 --master_address <MASTER_ADDRESS> \
--uni 1 --config-file projects/UNINEXT/configs/image_joint_r50.yaml \
--resume OUTPUT_DIR ./image_joint_r50
```

`<MASTER_ADDRESS>` should be the IP address of node 0. `<PORT>` should be the same among multiple nodes. If `<PORT>` is not specifed, programm will generate a random number as `<PORT>`.