import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
def inference_model(model, img, classes):
    """Inference image(s) with the classifier.
    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(img))
    input = image_transforms(img)
    input = input.unsqueeze(0)
    # forward the model
    with torch.no_grad():
        model.eval()
        scores = model(input)
        scores = torch.nn.functional.softmax(scores, dim=1)
        results = []
        for i in range(5):
            pred_score = torch.max(scores)
            pred_label = torch.argmax(scores).item()
            result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
            result['pred_class'] = classes[result['pred_label']]
            results.append(result)
            scores[0][pred_label] = -999999
    return results