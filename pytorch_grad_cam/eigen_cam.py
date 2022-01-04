from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
<<<<<<< HEAD
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None, return_model_output=True):
        super(EigenCAM, self).__init__(model, target_layer, use_cuda, 
            reshape_transform, return_model_output)
=======
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layers, use_cuda,
                                       reshape_transform)
>>>>>>> a3d5c27a4fc2b78faef5729e0953770969bb8ecd

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
