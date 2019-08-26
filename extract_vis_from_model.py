import torch
import utils

model = torch.load("model_supervised.pt", map_location="cpu")

visualization_filename = "visualization_supervised_all"
    # Reset
open(visualization_filename, 'w').close()

for layer in model.convnet:
    if isinstance(layer, torch.nn.Conv2d):
        for filter in layer.weight:
            filter = utils.normalize_01(filter)
            utils.save_image_visualization(filter.detach().cpu().numpy(),
                                           filename=visualization_filename)
