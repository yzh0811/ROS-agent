from dataclasses import dataclass
from pathlib import Path
import toml
from typing import Dict, Optional, Type, Any
from threading import RLock
from src.logger.logger import logger

import os

from src.paths import CONFIG_DIR


# ------------------- 基础配置类 -------------------
@dataclass
class BaseConfig:
    """所有配置类的基类"""
    pass


# ------------------- 具体配置类 -------------------

@dataclass
class ModelSelectConfig(BaseConfig):
    chat_model: str
    reasoner_model: str
    embedding_model: str
    chat_model_provider: str
    reasoner_model_provider: str
    embedding_model_provider: str


@dataclass
class LLMConfig(BaseConfig):
    model_name: str
    model_key: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float


@dataclass
class RegisterConfig(BaseConfig):
    host: str
    port: str
    namespace: str
    service_name: str
    group_name: int
    heartbeat_interval: float
    username: str
    password: str

@dataclass
class AppConfig(BaseConfig):
    app_host: str
    app_port: str


# ------------------- 配置管理器 -------------------
class ConfigManager:
    _instance = None
    _instance_lock = RLock()

    # 配置类型注册表（可扩展）
    CONFIG_REGISTRY = {
        "models": {
            "config_class": LLMConfig,
            "required_fields": ["model_name", "model_key", "base_url", "api_key", "max_tokens", "temperature"]
        },
        "defaultmodels": {
            "config_class": ModelSelectConfig,
            "required_fields": ["chat_model", "reasoner_model", "embedding_model", "embedding_model_provider",
                                "chat_model_provider", "reasoner_model_provider"]
        },
        "register": {
            "config_class": RegisterConfig,
            "required_fields": ["host", "port", "namespace", "service_name", "group_name", "heartbeat_interval", "username", "password"]
        },
        "service": {
            "config_class": AppConfig,
            "required_fields": ["app_host", "app_port"]
        },
    }

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        with self._instance_lock:
            if getattr(self, '_initialized', False):
                return
            try:
                logger.info("Initializing Config Manager...")

                # 使用嵌套字典存储配置 {section: {name: config}}
                self.configs: Dict[str, Dict[str, BaseConfig]] = {}
                self._config_path: Optional[Path] = None

                # 初始化锁
                self._config_lock = RLock()
                self._file_io_lock = RLock()

                # 加载默认配置
                default_path = CONFIG_DIR / "config.toml"
                if not self._safe_load_config(default_path):
                    raise RuntimeError("Failed to load default config")

                self._initialized = True
                logger.debug(f"Initialization completed. Loaded sections: {list(self.configs.keys())}")

            except Exception as e:
                logger.critical(f"Initialization failed: {str(e)}", exc_info=True)
                raise

    def _resolve_env_vars(self, config_dict: dict) -> dict:
        """解析环境变量占位符"""
        resolved = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                resolved_value = os.getenv(env_var)
                if not resolved_value:
                    raise ValueError(f"Environment variable {env_var} not set")
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        return resolved

    def _safe_load_config(self, config_path: Path) -> bool:
        """线程安全的配置加载核心逻辑"""
        try:
            with self._file_io_lock:
                with open(config_path, "r", encoding='utf-8') as f:
                    config_data = toml.load(f)
            new_configs = {}
            # 遍历所有配置块
            # 此时 section 的值就是这些顶层键
            for section, section_content in config_data.items():
                # 跳过未注册的配置块
                if section not in self.CONFIG_REGISTRY:
                    logger.warning(f"Ignoring unregistered config section: {section}")
                    continue

                registry_info = self.CONFIG_REGISTRY[section]
                config_class = registry_info["config_class"]
                required_fields = registry_info["required_fields"]
                section_configs = {}

                # 处理每个配置项
                for name, params in section_content.items():
                    try:
                        resolved_params = self._resolve_env_vars(params)

                        # 检查必填字段
                        missing = [f for f in required_fields if f not in resolved_params]
                        if missing:
                            logger.error(f"Config [{section}.{name}] missing required fields: {missing}")
                            continue

                        # 实例化配置对象
                        config_obj = config_class(**resolved_params)
                        section_configs[name] = config_obj

                    except Exception as e:
                        logger.error(f"Failed to create {section}.{name}: {str(e)}")
                        continue

                if section_configs:
                    new_configs[section] = section_configs
                    logger.info(f"Loaded {len(section_configs)} config(s) in section [{section}]")

            # 原子性更新配置
            with self._config_lock:
                self.configs = new_configs
                self._config_path = config_path

            return True

        except Exception as e:
            logger.error(f"Config loading failed: {str(e)}")
            return False

    def get_config(self, section: str, name: str) -> Optional[BaseConfig]:
        """通用配置获取方法"""
        with self._config_lock:
            _config = self.configs.get(section, {}).get(name)
            return _config

    def get_model_config(self, name: str) -> Optional[LLMConfig]:
        """获取模型配置（类型安全的快捷方式）"""
        return self.get_config("models", name)

    def get_model_select_config(self, name: str) -> Optional[ModelSelectConfig]:
        models_select = self.get_config("defaultmodels", name)
        return models_select

    def get_register_config(self, name: str) -> Optional[RegisterConfig]:
        models_select = self.get_config("register", name)
        return models_select

    def get_service_config(self, name: str) -> Optional[AppConfig]:
        service_select = self.get_config("service", name)
        return service_select


    def get_available_sections(self) -> list:
        """获取已加载的配置块列表"""
        with self._config_lock:
            configs_list = list(self.configs.keys())
            return configs_list

    def __str__(self):
        return f"ConfigManager(sections={self.get_available_sections()})"


manager = ConfigManager()
