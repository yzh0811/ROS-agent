# src/paths.py
from pathlib import Path
from typing import Final

# 核心路径定义
_PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
CONFIG_DIR: Final[Path] = _PROJECT_ROOT / "src" / "configs"
LOG_DIR: Final[Path] = _PROJECT_ROOT / "src" / "logs"
DATA_DIR: Final[Path] = _PROJECT_ROOT / "src" / "data"


def get_project_root() -> Path:
    """获取项目根目录"""
    return _PROJECT_ROOT


def ensure_dirs_exist():
    """确保所有需要的目录都存在"""
    required_dirs = [CONFIG_DIR, LOG_DIR, DATA_DIR]
    for directory in required_dirs:
        directory.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    print(f"_PROJECT_ROOT:{_PROJECT_ROOT}")
    print(f"CONFIG_DIR:{CONFIG_DIR}")
    print(f"LOG_DIR:{LOG_DIR}")
    print(f"DATA_DIR:{DATA_DIR}")
