import os
import sys
import yaml
import torch
import argparse

# 添加上一级目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.light_person_net import LightPersonIDNet, MobileFaceNet


def export_models(config_path="config.yaml", output_prefix="weights/best_model"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights_path = cfg["paths"]["best_model_path"]
    model_type = cfg["system"].get("model_type", "LightPersonIDNet")
    img_size = tuple(cfg["system"].get("image_size", [256, 128]))
    embedding_dim = cfg["system"].get("embedding_dim", 512)

    print(f"📦 开始导出轻量级模型: {model_type}")
    print(f"标准输入尺寸: (1, 3, {img_size[0]}, {img_size[1]})")

    if model_type == "MobileFaceNet":
        model = MobileFaceNet(embedding_dim=embedding_dim)
    else:
        model = LightPersonIDNet(embedding_dim=embedding_dim, pretrained=False)

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict) and "backbone" in ckpt:
            model.load_state_dict(ckpt["backbone"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        print(f"已加载权重: {weights_path}")
    else:
        print(f"⚠️ 权重文件 {weights_path} 不存在，导出基础结构")

    model.eval()
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)

    # 1. 导出 TorchScript 极速部署格式 (.pt)
    ts_path = f"{output_prefix}_traced.pt"
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        size_mb = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"✅ 成功导出 TorchScript 独立推理模型: {ts_path} ({size_mb:.2f} MB，可原生集成于 C++/Python 极速推理)")
    except Exception as e:
        print(f"❌ TorchScript 导出失败: {e}")

    # 2. 导出 ONNX 格式 (.onnx)
    onnx_path = f"{output_prefix}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            dynamo=False
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✅ 成功导出 ONNX 模型: {onnx_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"ℹ️ ONNX 导出跳过 (若需要 ONNX 格式请运行 pip install onnx): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 PyTorch 模型至 TorchScript/ONNX 极速部署格式")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--prefix", default="weights/best_model", help="输出文件前缀")
    args = parser.parse_args()

    cfg_file = args.config
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

    export_models(cfg_file, args.prefix)
