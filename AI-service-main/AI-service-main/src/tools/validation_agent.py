import json
import logging
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.tools.data_validator import data_validator
from src.configs.model_init import chat_model

logger = logging.getLogger(__name__)

class ValidationAgent:
    """数据验证代理，负责协调和执行数据验证流程"""
    
    def __init__(self, llm=None):
        self.llm = llm or chat_model
        self.validation_history = []
        
    def validate_excel_data(self, excel_data: Dict[str, Any], field_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        执行完整的Excel数据验证流程
        
        Args:
            excel_data: Excel数据
            field_mapping: 字段映射（可选）
            
        Returns:
            完整的验证结果
        """
        try:
            logger.info("开始执行数据验证流程")
            
            validation_result = {
                "overall_valid": True,
                "validation_steps": [],
                "summary": {},
                "recommendations": []
            }
            
            # 步骤1: 验证数据结构
            logger.info("步骤1: 验证数据结构")
            structure_validation = data_validator.validate_excel_data_structure(excel_data)
            validation_result["validation_steps"].append({
                "step": "structure_validation",
                "result": structure_validation
            })
            
            if not structure_validation["is_valid"]:
                validation_result["overall_valid"] = False
                validation_result["recommendations"].append("数据结构验证失败，请检查数据格式")
            
            # 步骤2: 验证字段映射（如果提供）
            if field_mapping:
                logger.info("步骤2: 验证字段映射")
                standard_fields = [
                    "运输日期", "订单号", "路顺", "承运商", "运单号", "车型", 
                    "发货方编号", "发货方名称", "收货方编码", "收货方名称", 
                    "商品编码", "商品名称", "件数", "体积", "重量"
                ]
                mapping_validation = data_validator.validate_field_mapping(field_mapping, standard_fields)
                validation_result["validation_steps"].append({
                    "step": "mapping_validation",
                    "result": mapping_validation
                })
                
                if not mapping_validation["is_valid"]:
                    validation_result["overall_valid"] = False
                    validation_result["recommendations"].append("字段映射验证失败，请检查映射关系")
            
            # 步骤3: 验证数据内容
            logger.info("步骤3: 验证数据内容")
            content_validation = data_validator.validate_data_content(excel_data, field_mapping or {})
            validation_result["validation_steps"].append({
                "step": "content_validation",
                "result": content_validation
            })
            
            if not content_validation["is_valid"]:
                validation_result["overall_valid"] = False
                validation_result["recommendations"].append("数据内容验证失败，请检查数据质量")
            
            # 步骤4: 生成验证摘要
            validation_result["summary"] = self._generate_validation_summary(validation_result["validation_steps"])
            
            # 步骤5: 使用LLM生成改进建议
            if self.llm:
                logger.info("步骤4: 生成AI改进建议")
                ai_recommendations = self._generate_ai_recommendations(validation_result)
                validation_result["ai_recommendations"] = ai_recommendations
            
            # 记录验证历史
            self.validation_history.append({
                "timestamp": self._get_current_timestamp(),
                "excel_filename": excel_data.get("filename", "unknown"),
                "result": validation_result
            })
            
            logger.info(f"数据验证完成，整体结果: {'通过' if validation_result['overall_valid'] else '失败'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"数据验证流程执行失败: {str(e)}")
            return {
                "overall_valid": False,
                "validation_steps": [],
                "summary": {"error": str(e)},
                "recommendations": [f"验证过程出错: {str(e)}"]
            }
    
    def validate_field_mapping_with_llm(self, excel_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """
        使用LLM响应验证字段映射
        
        Args:
            excel_data: Excel数据
            llm_response: LLM的字段映射响应
            
        Returns:
            验证结果
        """
        try:
            # 尝试解析LLM响应中的映射信息
            mapping = self._extract_mapping_from_llm_response(llm_response)
            
            if not mapping:
                return {
                    "is_valid": False,
                    "errors": ["无法从LLM响应中提取字段映射"],
                    "mapping": {},
                    "llm_response": llm_response
                }
            
            # 验证映射
            standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型", 
                "发货方编号", "发货方名称", "收货方编码", "收货方名称", 
                "商品编码", "商品名称", "件数", "体积", "重量"
            ]
            
            validation_result = data_validator.validate_field_mapping(mapping, standard_fields)
            validation_result["mapping"] = mapping
            validation_result["llm_response"] = llm_response
            
            return validation_result
            
        except Exception as e:
            logger.error(f"LLM字段映射验证失败: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"LLM映射验证过程出错: {str(e)}"],
                "mapping": {},
                "llm_response": llm_response
            }
    
    def _generate_validation_summary(self, validation_steps: List[Dict]) -> Dict[str, Any]:
        """生成验证摘要"""
        summary = {
            "total_steps": len(validation_steps),
            "passed_steps": 0,
            "failed_steps": 0,
            "quality_scores": {},
            "error_count": 0,
            "warning_count": 0
        }
        
        for step in validation_steps:
            step_result = step["result"]
            
            if step_result.get("is_valid", False):
                summary["passed_steps"] += 1
            else:
                summary["failed_steps"] += 1
            
            # 收集质量分数
            for key in ["data_quality_score", "mapping_quality_score", "content_quality_score"]:
                if key in step_result:
                    summary["quality_scores"][key] = step_result[key]
            
            # 统计错误和警告
            summary["error_count"] += len(step_result.get("errors", []))
            summary["warning_count"] += len(step_result.get("warnings", []))
        
        return summary
    
    def _generate_ai_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """使用LLM生成改进建议"""
        try:
            # 构建提示词
            prompt = self._build_recommendation_prompt(validation_result)
            
            # 调用LLM
            messages = [
                SystemMessage(content="你是一个数据质量专家，专门分析数据验证结果并提供改进建议。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # 解析响应
            recommendations = self._parse_ai_recommendations(response.content)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成AI建议失败: {str(e)}")
            return ["AI建议生成失败，请手动检查数据"]
    
    def _build_recommendation_prompt(self, validation_result: Dict[str, Any]) -> str:
        """构建AI建议提示词"""
        prompt = f"""
请分析以下数据验证结果，并提供具体的改进建议：

验证摘要：
- 总步骤数: {validation_result['summary'].get('total_steps', 0)}
- 通过步骤: {validation_result['summary'].get('passed_steps', 0)}
- 失败步骤: {validation_result['summary'].get('failed_steps', 0)}
- 错误数量: {validation_result['summary'].get('error_count', 0)}
- 警告数量: {validation_result['summary'].get('warning_count', 0)}
- 质量分数: {validation_result['summary'].get('quality_scores', {})}

验证步骤详情：
"""
        
        for step in validation_result.get("validation_steps", []):
            step_name = step["step"]
            step_result = step["result"]
            
            prompt += f"\n{step_name}:\n"
            prompt += f"- 是否通过: {'是' if step_result.get('is_valid', False) else '否'}\n"
            
            if step_result.get("errors"):
                prompt += f"- 错误: {step_result['errors']}\n"
            
            if step_result.get("warnings"):
                prompt += f"- 警告: {step_result['warnings']}\n"
        
        prompt += """
请提供3-5条具体的改进建议，每条建议应该：
1. 针对具体的验证问题
2. 提供可操作的解决方案
3. 按优先级排序
4. 使用中文回答

建议格式：
1. [优先级] 具体建议
2. [优先级] 具体建议
...
"""
        
        return prompt
    
    def _parse_ai_recommendations(self, ai_response: str) -> List[str]:
        """解析AI建议响应"""
        try:
            # 简单的行分割解析
            lines = ai_response.strip().split('\n')
            recommendations = []
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.') or 
                           line.startswith('5.') or line.startswith('-')):
                    # 移除序号和符号
                    clean_line = line.lstrip('1234567890.- ').strip()
                    if clean_line:
                        recommendations.append(clean_line)
            
            return recommendations if recommendations else ["请根据验证结果手动检查数据质量"]
            
        except Exception as e:
            logger.error(f"解析AI建议失败: {str(e)}")
            return ["AI建议解析失败，请手动检查数据"]
    
    def _extract_mapping_from_llm_response(self, llm_response: str) -> Dict[str, str]:
        """从LLM响应中提取字段映射"""
        try:
            # 尝试解析JSON格式的响应
            if "mapping" in llm_response.lower():
                # 查找JSON部分
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = llm_response[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    
                    if "mapping" in parsed:
                        return parsed["mapping"]
            
            # 如果无法解析，返回空字典
            return {}
            
        except Exception as e:
            logger.error(f"提取字段映射失败: {str(e)}")
            return {}
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_validation_history(self) -> List[Dict]:
        """获取验证历史"""
        return self.validation_history
    
    def clear_validation_history(self):
        """清除验证历史"""
        self.validation_history = []

# 创建全局实例
validation_agent = ValidationAgent() 