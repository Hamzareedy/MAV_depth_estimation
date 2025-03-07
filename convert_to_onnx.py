import torch
import config
import model
from torch.quantization import quantize_dynamic

def export_to_onnx(model_path, model_id, use_quantization=False):
    config.config["device"] = "cpu" # Always set device to cpu for export, so you don't get errors

    depth_model = model.DepthModel()
    depth_model.load_state_dict(torch.load(model_path))
    depth_model.eval()

    # Optimize for embedded CPU's by reducing precision (32 bit float to 8 bit int) to reduce 
    # model size and improve inference speed
    if use_quantization:
        depth_model = quantize_dynamic(depth_model, dtype=torch.qint8)

    # Export to ONNX
    # TODO: Infer the image height and width somehow, don't hardcode it
    dummy_input = torch.randn(1, config.config["input_channels"], 520, 240)

    onnx_model_path = config.config["save_model_path"] + f"/onnx_models/model_{model_id}.onnx"
    torch.onnx.export(depth_model, dummy_input, onnx_model_path)

    print(f"Exported model_{model_id} to ONNX in {onnx_model_path}")

if __name__ == "__main__":
    model_id = 0
    model_path = config.config["save_model_path"] + f"/model_{model_id}.pth"

    export_to_onnx(model_path, model_id, True)