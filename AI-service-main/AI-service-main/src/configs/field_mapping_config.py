import toml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FieldMappingConfig:
    """字段映射配置类"""
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    retry_delay: float = 1.0
    enable_logging: bool = True
    save_intermediate_results: bool = False
    standard_fields: List[str] = None
    field_priority: Dict[str, int] = None
    validation_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.standard_fields is None:
            self.standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
                "发货方编号", "发货方名称", "收货方编码", "收货方名称",
                "商品编码", "商品名称", "件数", "体积", "重量"
            ]
        
        if self.field_priority is None:
            self.field_priority = {
                "订单号": 1,
                "运单号": 2,
                "运输日期": 3,
                "件数": 4,
                "体积": 5,
                "重量": 6
            }
        
        if self.validation_rules is None:
            self.validation_rules = {
                "required_fields": ["订单号", "运单号"],
                "date_fields": ["运输日期"],
                "numeric_fields": ["件数", "体积", "重量"]
            }


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "src/configs/config.toml"):
        self.config_path = Path(config_path)
        self.config = None
        self.field_mapping_config = None
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                self.config = toml.load(self.config_path)
                self._parse_field_mapping_config()
                logger.info("配置文件加载成功")
            else:
                logger.warning(f"配置文件不存在: {self.config_path}")
                self._create_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            self._create_default_config()
    
    def _parse_field_mapping_config(self):
        """解析字段映射配置"""
        try:
            field_config = self.config.get("field_mapping", {})
            self.field_mapping_config = FieldMappingConfig(
                max_iterations=field_config.get("max_iterations", 3),
                confidence_threshold=field_config.get("confidence_threshold", 0.8),
                retry_delay=field_config.get("retry_delay", 1.0),
                enable_logging=field_config.get("enable_logging", True),
                save_intermediate_results=field_config.get("save_intermediate_results", False),
                standard_fields=field_config.get("standard_fields", []),
                field_priority=field_config.get("field_priority", {}),
                validation_rules=field_config.get("validation_rules", {})
            )
        except Exception as e:
            logger.error(f"解析字段映射配置失败: {str(e)}")
            self.field_mapping_config = FieldMappingConfig()
    
    def _create_default_config(self):
        """创建默认配置"""
        logger.info("使用默认配置")
        self.field_mapping_config = FieldMappingConfig()
    
    def get_field_mapping_config(self) -> FieldMappingConfig:
        """获取字段映射配置"""
        return self.field_mapping_config
    
    def update_config(self, **kwargs):
        """更新配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.field_mapping_config, key):
                    setattr(self.field_mapping_config, key, value)
                    logger.info(f"配置已更新: {key} = {value}")
                else:
                    logger.warning(f"未知配置项: {key}")
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")
    
    def reload_config(self):
        """重新加载配置"""
        logger.info("重新加载配置文件")
        self._load_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        if self.field_mapping_config:
            return {
                "max_iterations": self.field_mapping_config.max_iterations,
                "confidence_threshold": self.field_mapping_config.confidence_threshold,
                "standard_fields_count": len(self.field_mapping_config.standard_fields),
                "field_priority_count": len(self.field_mapping_config.field_priority),
                "validation_rules_count": len(self.field_mapping_config.validation_rules)
            }
        return {}


# 全局配置管理器实例
config_manager = ConfigManager() 