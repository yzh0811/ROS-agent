import json
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器，提供多种数据验证功能"""
    
    def __init__(self):
        self.validation_rules = {
            "required_fields": [],
            "data_types": {},
            "value_ranges": {},
            "format_patterns": {},
            "custom_validators": {}
        }
    
    def validate_excel_data_structure(self, excel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证Excel数据结构
        
        Args:
            excel_data: Excel数据字典
            
        Returns:
            验证结果
        """
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "data_quality_score": 0,
                "validation_details": {}
            }
            
            # 检查基本结构
            required_keys = ["filename", "total_rows", "columns", "data"]
            for key in required_keys:
                if key not in excel_data:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"缺少必需字段: {key}")
            
            if not validation_result["is_valid"]:
                return validation_result
            
            # 检查数据类型
            if not isinstance(excel_data["total_rows"], int) or excel_data["total_rows"] < 0:
                validation_result["errors"].append("total_rows必须是正整数")
                validation_result["is_valid"] = False
            
            if not isinstance(excel_data["columns"], list):
                validation_result["errors"].append("columns必须是列表")
                validation_result["is_valid"] = False
            
            if not isinstance(excel_data["data"], list):
                validation_result["errors"].append("data必须是列表")
                validation_result["is_valid"] = False
            
            # 检查数据一致性
            if excel_data["data"]:
                expected_columns = set(excel_data["columns"])
                for i, row in enumerate(excel_data["data"]):
                    if not isinstance(row, dict):
                        validation_result["errors"].append(f"第{i+1}行数据不是字典格式")
                        validation_result["is_valid"] = False
                        continue
                    
                    row_columns = set(row.keys())
                    if row_columns != expected_columns:
                        missing_cols = expected_columns - row_columns
                        extra_cols = row_columns - expected_columns
                        if missing_cols:
                            validation_result["warnings"].append(f"第{i+1}行缺少列: {missing_cols}")
                        if extra_cols:
                            validation_result["warnings"].append(f"第{i+1}行多余列: {extra_cols}")
            
            # 计算数据质量分数
            validation_result["data_quality_score"] = self._calculate_quality_score(
                excel_data, validation_result
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"数据验证失败: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"验证过程出错: {str(e)}"],
                "warnings": [],
                "data_quality_score": 0,
                "validation_details": {}
            }
    
    def validate_field_mapping(self, mapping: Dict[str, str], standard_fields: List[str]) -> Dict[str, Any]:
        """
        验证字段映射的合理性
        
        Args:
            mapping: 字段映射字典
            standard_fields: 标准字段列表
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "mapping_quality_score": 0,
            "validation_details": {}
        }
        
        try:
            # 检查映射完整性
            mapped_fields = set(mapping.values())
            mapped_fields.discard("missing")  # 排除"missing"标记
            
            unmapped_fields = set(standard_fields) - mapped_fields
            if unmapped_fields:
                validation_result["warnings"].append(f"未映射的标准字段: {unmapped_fields}")
            
            # 检查重复映射
            value_counts = {}
            for key, value in mapping.items():
                if value != "missing":
                    value_counts[value] = value_counts.get(value, 0) + 1
            
            duplicates = {value: count for value, count in value_counts.items() if count > 1}
            if duplicates:
                validation_result["errors"].append(f"重复映射的字段: {duplicates}")
                validation_result["is_valid"] = False
            
            # 检查映射合理性
            suspicious_mappings = []
            for std_field, custom_field in mapping.items():
                if custom_field != "missing":
                    similarity = self._calculate_field_similarity(std_field, custom_field)
                    if similarity < 0.3:  # 相似度阈值
                        suspicious_mappings.append({
                            "standard_field": std_field,
                            "custom_field": custom_field,
                            "similarity": similarity
                        })
            
            if suspicious_mappings:
                validation_result["warnings"].append(f"可疑的字段映射: {suspicious_mappings}")
            
            # 计算映射质量分数
            validation_result["mapping_quality_score"] = self._calculate_mapping_quality(
                mapping, standard_fields, validation_result
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"字段映射验证失败: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"映射验证过程出错: {str(e)}"],
                "warnings": [],
                "mapping_quality_score": 0,
                "validation_details": {}
            }
    
    def validate_data_content(self, excel_data: Dict[str, Any], field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        验证数据内容的质量
        
        Args:
            excel_data: Excel数据
            field_mapping: 字段映射
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "content_quality_score": 0,
            "validation_details": {}
        }
        
        try:
            if not excel_data.get("data"):
                validation_result["errors"].append("没有数据可验证")
                validation_result["is_valid"] = False
                return validation_result
            
            data = excel_data["data"]
            columns = excel_data["columns"]
            
            # 检查空值率
            null_rates = {}
            for col in columns:
                null_count = sum(1 for row in data if row.get(col) is None or row.get(col) == "")
                null_rate = null_count / len(data)
                null_rates[col] = null_rate
                
                if null_rate > 0.5:  # 空值率超过50%
                    validation_result["warnings"].append(f"列 '{col}' 空值率过高: {null_rate:.2%}")
            
            # 检查数据类型一致性
            type_inconsistencies = {}
            for col in columns:
                types = set()
                for row in data:
                    if row.get(col) is not None:
                        types.add(type(row[col]).__name__)
                
                if len(types) > 1:
                    type_inconsistencies[col] = list(types)
                    validation_result["warnings"].append(f"列 '{col}' 数据类型不一致: {types}")
            
            # 检查异常值
            numeric_columns = []
            for col in columns:
                if any(isinstance(row.get(col), (int, float)) for row in data if row.get(col) is not None):
                    numeric_columns.append(col)
            
            for col in numeric_columns:
                values = [row[col] for row in data if isinstance(row.get(col), (int, float))]
                if values:
                    mean_val = sum(values) / len(values)
                    std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                    
                    outliers = [v for v in values if abs(v - mean_val) > 3 * std_val]
                    if outliers:
                        validation_result["warnings"].append(f"列 '{col}' 存在异常值: {outliers[:5]}...")
            
            validation_result["validation_details"] = {
                "null_rates": null_rates,
                "type_inconsistencies": type_inconsistencies,
                "numeric_columns": numeric_columns
            }
            
            # 计算内容质量分数
            validation_result["content_quality_score"] = self._calculate_content_quality(
                data, columns, validation_result
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"数据内容验证失败: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"内容验证过程出错: {str(e)}"],
                "warnings": [],
                "content_quality_score": 0,
                "validation_details": {}
            }
    
    def _calculate_quality_score(self, excel_data: Dict[str, Any], validation_result: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        score = 100.0
        
        # 根据错误数量扣分
        score -= len(validation_result["errors"]) * 20
        
        # 根据警告数量扣分
        score -= len(validation_result["warnings"]) * 5
        
        # 根据数据行数调整分数
        if excel_data.get("total_rows", 0) == 0:
            score = 0
        elif excel_data.get("total_rows", 0) < 10:
            score *= 0.8
        
        return max(0, min(100, score))
    
    def _calculate_mapping_quality(self, mapping: Dict[str, str], standard_fields: List[str], validation_result: Dict[str, Any]) -> float:
        """计算映射质量分数"""
        score = 100.0
        
        # 根据错误数量扣分
        score -= len(validation_result["errors"]) * 30
        
        # 根据警告数量扣分
        score -= len(validation_result["warnings"]) * 10
        
        # 根据映射完整性调整分数
        mapped_count = sum(1 for v in mapping.values() if v != "missing")
        completeness = mapped_count / len(standard_fields)
        score *= completeness
        
        return max(0, min(100, score))
    
    def _calculate_content_quality(self, data: List[Dict], columns: List[str], validation_result: Dict[str, Any]) -> float:
        """计算内容质量分数"""
        score = 100.0
        
        # 根据错误数量扣分
        score -= len(validation_result["errors"]) * 25
        
        # 根据警告数量扣分
        score -= len(validation_result["warnings"]) * 8
        
        # 根据空值率调整分数
        if validation_result.get("validation_details", {}).get("null_rates"):
            avg_null_rate = sum(validation_result["validation_details"]["null_rates"].values()) / len(columns)
            score *= (1 - avg_null_rate)
        
        return max(0, min(100, score))
    
    def _calculate_field_similarity(self, field1: str, field2: str) -> float:
        """计算字段相似度"""
        # 简单的字符串相似度计算
        field1_lower = field1.lower()
        field2_lower = field2.lower()
        
        # 检查包含关系
        if field1_lower in field2_lower or field2_lower in field1_lower:
            return 0.8
        
        # 检查字符重叠
        common_chars = set(field1_lower) & set(field2_lower)
        total_chars = set(field1_lower) | set(field2_lower)
        
        if total_chars:
            return len(common_chars) / len(total_chars)
        
        return 0.0

# 创建全局实例
data_validator = DataValidator() 