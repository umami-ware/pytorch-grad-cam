import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/1710.11063


class GradCAMPlusPlus(BaseCAM):
<<<<<<< HEAD
    def __init__(self, model, target_layer, use_cuda=False,
        reshape_transform=None, return_model_output=True):
        super(GradCAMPlusPlus, self).__init__(model, target_layer, use_cuda, 
            reshape_transform, return_model_output)
=======
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform)
>>>>>>> a3d5c27a4fc2b78faef5729e0953770969bb8ecd

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights
