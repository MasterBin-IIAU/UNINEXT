# UNINEXT MODEL ZOO

## Introduction
UNINEXT achieves superior performance on 20 benchmarks, using the same model with the same model parameters. UNINEXT has 3 training stages, pretraining, image-level joint training, and video-level joint training. We provide all the checkpoints of all stages for models with different backbones.

### Stage 1: Pretraining

<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>obj365v2_32g_r50</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EramwIArPfVDstllO1TCXWcB3L2ZHeD6X87RtJ0k3HPZ9w?e=qcDrgf">model</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-Large</td>
    <td>obj365v2_32g_convnext_large</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Ei8uhzoVZ1pCuxLUcvDzficBz86JYSz4G43cv8V1Yaht5A?e=kydXcv">model</a></td>
  </tr>
  <tr>
    <td>ViT-Huge</td>
    <td>obj365v2_32g_vit_huge</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Es3slaW09a5El6lM2UU5fzsBpEwzzwDnhJtreZpVhrrxrA?e=LUs3vd">model</a></td>
  </tr>
</table>

### Stage 2: Image-level Joint Training

<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>image_joint_r50</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EkfdtpEnPbZEjUToUGfJ_GMBXPRAPro27hc-tk40PUD8VA?e=8oIKkr">model</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-Large</td>
    <td>image_joint_convnext_large</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Et6GBDgKgPZDn5zp49yKwDYBd50EBTxaKs7R6Yuck_lf7g?e=818rMm">model</a></td>
  </tr>
  <tr>
    <td>ViT-Huge</td>
    <td>image_joint_vit_huge_32g</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/ElhVBgRJRKhLu-2xeQliAj8Bq4F1fo83ZLnodi_YEAEB3Q?e=JYInoo">model</a></td>
  </tr>
</table>

### Stage 3: Video-level Joint Training
All numbers reported in the paper (Table 1 to Table 10) uses the following models.
<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>video_joint_r50</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/ErbTZCzv0vJAoIMwa90_3qoBOFbHIJJTVxI58-kk2nfkhw?e=4qvjrR">model</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-Large</td>
    <td>video_joint_convnext_large</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EiVn7fSMVq9CroNvMIbosUsBhNpLNn7E0tmLqJlDL6xcoQ?e=u6YUNu">model</a></td>
  </tr>
  <tr>
    <td>ViT-Huge</td>
    <td>video_joint_vit_huge</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Eoe3Z5YAwi1Mj5gL_jJnXcEBuMnxPEM9yWtjE-pERp6mkg?e=rFbyVi">model</a></td>
  </tr>
</table>

Please note that the pretrained weights used in this stage ends with `model_final_4c.pth`. To obtain these weights, please run the following commands

```
python3 conversion/convert_3c_to_4c_pth.py # ResNet backbone
python3 conversion/convert_3c_to_4c_pth_convnext.py # ConvNeXt backbone
python3 conversion/convert_3c_to_4c_pth_vit.py # ViT backbone
```

### Single Tasks
We also provide models trained on a single task with ResNet-50 backbone (Table 11 in the paper).
<table>
  <tr>
    <th>Task</th>
    <th>YAML</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>OD&IS</td>
    <td>single_task_det</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EvcywQKg-ytDt9KM5OGxUXYBDD95_letMYOqiAJ_x4RsrA?e=AAgZL8">model</a></td>
  </tr>
  <tr>
    <td>REC&RES</td>
    <td>single_task_rec</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EhmyBlRME9hMp-Go-SPJs9kBXhJ83lLryw-JNOuEl0Ntdw?e=Ilt3M2">model</a></td>
  </tr>
  <tr>
    <td>VIS</td>
    <td>single_task_vis</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Eu0MquVcxWBNlHBTQArWUREB-qqIjqtmYqlGQJAGLvqHHg?e=o8sX21">model</a></td>
  </tr>
    <td>RVOS</td>
    <td>single_task_rvos</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Eo5OwH6aRLNKvUhMZkGjGBEBpPrXBJPJ-Ym3XF516MAfkg?e=6eTDGL">model</a></td>
  </tr>
    <td>SOT&VOS</td>
    <td>single_task_sot</td>
    <td><a href="https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Eih6E00EDahKsiajD-yjkhkBRnuO1Tg6ZsyVM3I8EHeDGw?e=7wfkp1">model</a></td>
  </tr>
</table>