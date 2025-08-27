import json
import logging
from typing import Dict, List, Any, Optional
from src.tools.data_validator import data_validator
from src.tools.validation_agent import validation_agent

logger = logging.getLogger(__name__)

# Function Calling 函数定义
VALIDATION_FUNCTIONS = [
    {
        "name": "validate_excel_structure",
        "description": "验证Excel数据的基本结构，包括必需字段、数据类型和数据一致性",
        "parameters": {
            "type": "object",
            "properties": {
                "excel_data": {
                    "type": "object",
                    "description": "Excel数据字典，包含filename、total_rows、columns、data等字段"
                },
                "excel_filename": {
                    "type": "string",
                    "description": "Excel文件名，系统会自动读取文件内容"
                }
            },
            "required": []
        }
    },
    {
        "name": "validate_field_mapping",
        "description": "验证字段映射的合理性，检查映射完整性、重复映射和映射质量",
        "parameters": {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "description": "字段映射字典，键为标准字段，值为自定义字段或'missing'"
                },
                "standard_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标准字段列表"
                }
            },
            "required": ["mapping", "standard_fields"]
        }
    },
    {
        "name": "validate_data_content",
        "description": "验证数据内容质量，包括空值率、数据类型一致性、异常值检测等",
        "parameters": {
            "type": "object",
            "properties": {
                "excel_data": {
                    "type": "object",
                    "description": "Excel数据字典"
                },
                "excel_filename": {
                    "type": "string",
                    "description": "Excel文件名，系统会自动读取文件内容"
                },
                "field_mapping": {
                    "type": "object",
                    "description": "字段映射字典（可选）"
                }
            },
            "required": []
        }
    },
    {
        "name": "comprehensive_validation",
        "description": "执行完整的数据验证流程，包括结构、映射和内容验证",
        "parameters": {
            "type": "object",
            "properties": {
                "excel_data": {
                    "type": "object",
                    "description": "Excel数据字典"
                },
                "excel_filename": {
                    "type": "string",
                    "description": "Excel文件名，系统会自动读取文件内容"
                },
                "field_mapping": {
                    "type": "object",
                    "description": "字段映射字典（可选）"
                },
                "include_ai_recommendations": {
                    "type": "boolean",
                    "description": "是否包含AI生成的改进建议",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "get_validation_summary",
        "description": "获取验证摘要，包括质量分数、错误统计和改进建议",
        "parameters": {
            "type": "object",
            "properties": {
                "validation_result": {
                    "type": "object",
                    "description": "验证结果字典"
                }
            },
            "required": ["validation_result"]
        }
    }
]

def validate_excel_structure(excel_data: Dict[str, Any] = None, excel_filename: str = None) -> Dict[str, Any]:
    """
    验证Excel数据的基本结构
    
    Args:
        excel_data: Excel数据字典
        excel_filename: Excel文件名
        
    Returns:
        验证结果
    """
    try:
        logger.info("执行Excel结构验证")
        
        # 如果提供了文件名，从文件读取数据
        if excel_filename and not excel_data:
            from src.utils.excel_processor import excel_processor
            excel_data = excel_processor.read_excel_to_json(excel_filename)
        
        if not excel_data:
            return {
                "success": False,
                "error": "缺少Excel数据或文件名",
                "message": "结构验证失败"
            }
        
        result = data_validator.validate_excel_data_structure(excel_data)
        return {
            "success": True,
            "result": result,
            "message": "结构验证完成"
        }
    except Exception as e:
        logger.error(f"Excel结构验证失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "结构验证失败"
        }

def validate_field_mapping(mapping: Dict[str, str], standard_fields: List[str]) -> Dict[str, Any]:
    """
    验证字段映射的合理性
    
    Args:
        mapping: 字段映射字典
        standard_fields: 标准字段列表
        
    Returns:
        验证结果
    """
    try:
        logger.info("执行字段映射验证")
        result = data_validator.validate_field_mapping(mapping, standard_fields)
        return {
            "success": True,
            "result": result,
            "message": "字段映射验证完成"
        }
    except Exception as e:
        logger.error(f"字段映射验证失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "字段映射验证失败"
        }

def validate_data_content(excel_data: Dict[str, Any] = None, excel_filename: str = None, field_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    验证数据内容质量
    
    Args:
        excel_data: Excel数据字典
        excel_filename: Excel文件名
        field_mapping: 字段映射字典（可选）
        
    Returns:
        验证结果
    """
    try:
        logger.info("执行数据内容验证")
        
        # 如果提供了文件名，从文件读取数据
        if excel_filename and not excel_data:
            from src.utils.excel_processor import excel_processor
            excel_data = excel_processor.read_excel_to_json(excel_filename)
        
        if not excel_data:
            return {
                "success": False,
                "error": "缺少Excel数据或文件名",
                "message": "数据内容验证失败"
            }
        
        result = data_validator.validate_data_content(excel_data, field_mapping or {})
        return {
            "success": True,
            "result": result,
            "message": "数据内容验证完成"
        }
    except Exception as e:
        logger.error(f"数据内容验证失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "数据内容验证失败"
        }

def comprehensive_validation(excel_data: Dict[str, Any] = None, excel_filename: str = None, field_mapping: Optional[Dict[str, str]] = None, include_ai_recommendations: bool = True) -> Dict[str, Any]:
    """
    执行完整的数据验证流程
    
    Args:
        excel_data: Excel数据字典
        excel_filename: Excel文件名
        field_mapping: 字段映射字典（可选）
        include_ai_recommendations: 是否包含AI建议
        
    Returns:
        完整的验证结果
    """
    try:
        logger.info("执行综合数据验证")
        
        # 如果提供了文件名，从文件读取数据
        if excel_filename and not excel_data:
            from src.utils.excel_processor import excel_processor
            excel_data = excel_processor.read_excel_to_json(excel_filename)
        
        if not excel_data:
            return {
                "success": False,
                "error": "缺少Excel数据或文件名",
                "message": "综合验证失败"
            }
        
        result = validation_agent.validate_excel_data(excel_data, field_mapping)
        
        # 如果不包含AI建议，移除相关字段
        if not include_ai_recommendations and "ai_recommendations" in result:
            del result["ai_recommendations"]
        
        return {
            "success": True,
            "result": result,
            "message": "综合验证完成"
        }
    except Exception as e:
        logger.error(f"综合数据验证失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "综合验证失败"
        }

def get_validation_summary(validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取验证摘要
    
    Args:
        validation_result: 验证结果字典
        
    Returns:
        验证摘要
    """
    try:
        logger.info("生成验证摘要")
        
        summary = {
            "overall_status": "通过" if validation_result.get("overall_valid", False) else "失败",
            "total_steps": validation_result.get("summary", {}).get("total_steps", 0),
            "passed_steps": validation_result.get("summary", {}).get("passed_steps", 0),
            "failed_steps": validation_result.get("summary", {}).get("failed_steps", 0),
            "error_count": validation_result.get("summary", {}).get("error_count", 0),
            "warning_count": validation_result.get("summary", {}).get("warning_count", 0),
            "quality_scores": validation_result.get("summary", {}).get("quality_scores", {}),
            "recommendations": validation_result.get("recommendations", []),
            "ai_recommendations": validation_result.get("ai_recommendations", [])
        }
        
        return {
            "success": True,
            "summary": summary,
            "message": "验证摘要生成完成"
        }
    except Exception as e:
        logger.error(f"生成验证摘要失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "验证摘要生成失败"
        }

# 函数映射字典
VALIDATION_FUNCTION_MAP = {
    "validate_excel_structure": validate_excel_structure,
    "validate_field_mapping": validate_field_mapping,
    "validate_data_content": validate_data_content,
    "comprehensive_validation": comprehensive_validation,
    "get_validation_summary": get_validation_summary
}

def execute_validation_function(function_name: str, **kwargs) -> Dict[str, Any]:
    """
    执行验证函数
    
    Args:
        function_name: 函数名称
        **kwargs: 函数参数
        
    Returns:
        函数执行结果
    """
    try:
        if function_name not in VALIDATION_FUNCTION_MAP:
            return {
                "success": False,
                "error": f"未知的验证函数: {function_name}",
                "message": "函数不存在"
            }
        
        function = VALIDATION_FUNCTION_MAP[function_name]
        result = function(**kwargs)
        
        return result
        
    except Exception as e:
        logger.error(f"执行验证函数失败: {function_name}, 错误: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"函数执行失败: {function_name}"
        }

def get_available_functions() -> List[Dict[str, Any]]:
    """
    获取可用的验证函数列表
    
    Returns:
        函数定义列表
    """
    return VALIDATION_FUNCTIONS

def get_function_schema(function_name: str) -> Optional[Dict[str, Any]]:
    """
    获取指定函数的schema
    
    Args:
        function_name: 函数名称
        
    Returns:
        函数schema
    """
    for func in VALIDATION_FUNCTIONS:
        if func["name"] == function_name:
            return func
    return None 