# src/__init__.py 或项目入口文件
from dotenv import load_dotenv
from src.paths import _PROJECT_ROOT
from src.logger.logger import logger


def init_env():
    # 从项目根目录加载.env文件
    env_path = _PROJECT_ROOT / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f"在{env_path}未找到.env文件，将使用系统环境变量")


# 在项目启动时调用
init_env()
