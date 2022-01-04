import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

# https://ieeexplore.ieee.org/document/9462463


class LayerCAM(BaseCAM):
<<<<<<< HEAD
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None,
    return_model_output=True):
        super(LayerCAM, self).__init__(model, target_layer, use_cuda, reshape_transform,
        return_model_output)
=======
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(
            LayerCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
>>>>>>> a3d5c27a4fc2b78faef5729e0953770969bb8ecd

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        spatial_weighted_activations = np.maximum(grads, 0) * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam
