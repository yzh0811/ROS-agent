import os
from langchain_openai import ChatOpenAI
from typing import Optional
import logging
import sys
import io
from src.configs.config import Config
from src.configs.settings import manager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# # 加载 .env 文件
# load_dotenv()
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
# DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL")
# LLM_TYPE = os.getenv("LLM_TYPE")
# TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE")
manager  = manager.get_model_config()
DEEPSEEK_API_KEY = Config.DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = Config.DEEPSEEK_BASE_URL
DEEPSEEK_CHAT_MODEL = Config.DEEPSEEK_CHAT_MODEL
LLM_TYPE = Config.LLM_TYPE
TEMPERATURE = Config.TEMPERATURE

# 默认配置
DEFAULT_LLM_TYPE = LLM_TYPE
DEFAULT_TEMPERATURE = TEMPERATURE

# 模型配置字典
MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://yunwu.ai/v1",
        "api_key": "sk-rmfPKCQYU7yWyX2RDideh1IggooRo8PVh8A42e3wL5zOFxKF",
        "model": "gpt-4o-mini"
    },
    "oneapi": {
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": "sk-ROhn6RNxulVXhlkZ0713F29093Ea49AcAcA29b96125aF1Ff",
        "model": "qwen-max"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-5cee351038c943648971907366eabafe",
        "model": "qwen-max"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "deepseek-r1:14b"
    },
    "deepseek": {
        "base_url": DEEPSEEK_BASE_URL,
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_CHAT_MODEL,
    }
}





class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> Optional[ChatOpenAI]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        ChatOpenAI: 初始化后的LLM实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理ollama类型
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        # 创建LLM实例
        llm = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=DEFAULT_TEMPERATURE,
            # 添加超时配置（秒）
            timeout=30,
            # 添加重试次数
            max_retries=2
        )
        # print(llm.invoke("hello world"))
        logger.info(f"成功初始化 {llm_type} LLM")
        return llm

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        ChatOpenAI: LLM实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise  # 如果默认配置也失败，则抛出异常


# 示例使用
if __name__ == "__main__":
    try:
        # 测试不同类型的LLM初始化
        # llm_openai = get_llm("openai")
        llm_qwen = get_llm("qwen")

        # 测试无效类型
        # llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")