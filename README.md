---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{}
---

# object-detection: detr-finetuned-thermal-dogs-and-people

<!-- Provide a quick summary of what the model is/does. -->

This model is a fine-tuned version of [DETR](https://huggingface.co/facebook/detr-resnet-50) on the Roboflow [Thermal Dogs and People](https://public.roboflow.com/object-detection/thermal-dogs-and-people/1) dataset.
It achieves the following results on the evaluation set:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.681
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.870
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.778
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.746
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794
```
## Intended purpose

Main purpose for this model are solely for learning purposes.

Thermal images have a wide array of applications: monitoring machine performance, seeing in low light conditions, and adding another dimension to standard RGB scenarios. Infrared imaging is useful in security, wildlife detection,and hunting / outdoors recreation.

## Training and evaluation data

Data can be seen at [Weights and Biases](https://wandb.ai/faldeus0092/thermal-dogs-and-people/runs/zjt8bp9x?workspace=user-faldeus0092)

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-4
- lr_backbone: 1e-5
- weight_decay: 1e-4
- optimizer: AdamW
- train_batch_size: 4
- eval_batch_size: 2
- train_set: 142
- test_set: 41
- num_epochs: 68

### Example usage (transformers pipeline)
```py
# Use a pipeline as a high-level helper
from transformers import pipeline

image = Image.open('/content/Thermal-Dogs-and-People-1/test/IMG_0006 5_jpg.rf.cd46e6a862d6ffb7fce6795067ce7cc7.jpg')
# image = Image.open(requests.get(url, stream=True).raw) # if you want to open from url

obj_detector = pipeline("object-detection", model="faldeus0092/detr-finetuned-thermal-dogs-and-people")

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

image
```


