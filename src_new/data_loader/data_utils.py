import os

def ensure_dir(path: str) -> str:
    """
    确保目录存在；若不存在则创建。
    """
    os.makedirs(path, exist_ok=True)
    return path
