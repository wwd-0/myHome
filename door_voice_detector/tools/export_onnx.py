"""
独立导出 ONNX 部署模型工具 (export_onnx.py)
支持从已训练的 PyTorch 权重 (.pt) 或基础结构一键导出至 ./audio/models/
"""

import os
import sys
import json
import argparse
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import BCResMDTCCommandNet, CAMPlusLiteSpeakerNet


def export_all(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    commands = cfg["commands"]["list"]
    num_commands = len(commands)
    s1_cfg = cfg["stage1_command_net"]
    s2_cfg = cfg["stage2_speaker_net"]

    weights_dir = cfg["paths"]["weights_dir"]
    export_dir = cfg["paths"]["export_dir"]
    os.makedirs(export_dir, exist_ok=True)

    cmd_net = BCResMDTCCommandNet(
        num_commands=num_commands,
        embed_dim=int(s1_cfg["embed_dim"]),
        base_channels=int(s1_cfg["base_channels"]),
        mdtc_channels=int(s1_cfg["mdtc_channels"]),
    )
    s1_ckpt = os.path.join(weights_dir, "best_command_net.pt")
    if os.path.exists(s1_ckpt):
        cmd_net.load_state_dict(torch.load(s1_ckpt, map_location="cpu"))
        print(f"✅ 已加载 Stage-1 口令网权重: {s1_ckpt}")
    else:
        cmd_net.normalize_classifier_weights()
        print(f"⚠️ 未找到 {s1_ckpt}，导出初始化结构权重")
    cmd_net.eval()

    spk_net = CAMPlusLiteSpeakerNet(
        embed_dim=int(s2_cfg["embed_dim"]),
        stem_channels=int(s2_cfg["stem_channels"]),
        init_channels=int(s2_cfg["init_channels"]),
        growth_rate=int(s2_cfg["growth_rate"]),
        num_cam_layers=int(s2_cfg["num_cam_layers"]),
    )
    s2_ckpt = os.path.join(weights_dir, "best_speaker_net.pt")
    if os.path.exists(s2_ckpt):
        spk_net.load_state_dict(torch.load(s2_ckpt, map_location="cpu"))
        print(f"✅ 已加载 Stage-2 声纹网权重: {s2_ckpt}")
    else:
        print(f"⚠️ 未找到 {s2_ckpt}，导出初始化结构权重")
    spk_net.eval()

    dummy = torch.randn(1, 1, 80, 160)
    kws_onnx = os.path.join(export_dir, cfg["paths"]["kws_onnx_name"])
    sv_onnx = os.path.join(export_dir, cfg["paths"]["sv_onnx_name"])
    cmd_json = os.path.join(export_dir, cfg["paths"]["commands_json_name"])

    torch.onnx.export(
        cmd_net, dummy, kws_onnx,
        opset_version=12, do_constant_folding=True,
        input_names=["fbank"], output_names=["output"], dynamo=False,
    )
    torch.onnx.export(
        spk_net, dummy, sv_onnx,
        opset_version=12, do_constant_folding=True,
        input_names=["fbank"], output_names=["spk_embedding"], dynamo=False,
    )

    if not os.path.exists(cmd_json):
        meta = {
            "commands": commands,
            "embed_dim": int(s1_cfg["embed_dim"]),
            "speaker_embed_dim": int(s2_cfg["embed_dim"]),
            "prototypes": {},
        }
        with open(cmd_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"📦 Stage-1 ONNX: {kws_onnx} ({os.path.getsize(kws_onnx)/1024/1024:.2f} MB)")
    print(f"📦 Stage-2 ONNX: {sv_onnx} ({os.path.getsize(sv_onnx)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"),
    )
    args = parser.parse_args()
    export_all(args.config)

