import torch
from models.mobilenet_pt import MobileNetv1

input_size = (3, 32, 32)
random_input_tensor = torch.randn(1, *input_size)

def conv():
    model = MobileNetv1()
    state = torch.load("mbn.pt", map_location=torch.device('cpu'))

    # Filter out unexpected keys from state_dict
    state = {k: v for k, v in state.items() if k in model.state_dict()}

    model.load_state_dict(state)
    torch.onnx.export(model, random_input_tensor, 'mbn_model_pt_rpi.onnx', export_params=True, opset_version=17)


conv()

