import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
<<<<<<< HEAD
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None, return_model_output=True):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform,
        return_model_output)
=======
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
>>>>>>> a3d5c27a4fc2b78faef5729e0953770969bb8ecd

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
