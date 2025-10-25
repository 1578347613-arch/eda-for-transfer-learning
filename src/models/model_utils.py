import torch
import config

def load_backbone_from_trained_mlp(pretrained_mlp, model):
    """
    从预训练 MLP 加载主干部分的权重
    """
    model.backbone.load_state_dict(pretrained_mlp.state_dict())
    print("已加载预训练 MLP 主干。")

def save_model(model, file_path):
    """
    保存模型权重
    """
    torch.save(model.state_dict(), file_path)
    print(f"模型已保存至: {file_path}")

def load_model(model, file_path):
    """
    从指定路径加载模型权重
    """
    model.load_state_dict(torch.load(file_path))
    print(f"模型已从 {file_path} 加载。")
