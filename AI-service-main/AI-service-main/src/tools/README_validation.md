# 数据验证功能说明

## 概述

本项目新增了完整的数据验证功能，支持多种验证方式：
- **Agent验证**：使用智能代理进行数据验证
- **Function验证**：通过函数调用进行验证
- **Function Calling**：支持LLM的函数调用验证
- **集成验证**：在聊天接口中集成验证功能

## 功能特性

### 1. 数据验证器 (DataValidator)

**位置**: `src/tools/data_validator.py`

**主要功能**:
- 验证Excel数据结构
- 验证字段映射合理性
- 验证数据内容质量
- 计算数据质量分数

**核心方法**:
```python
# 验证Excel数据结构
validate_excel_data_structure(excel_data)

# 验证字段映射
validate_field_mapping(mapping, standard_fields)

# 验证数据内容
validate_data_content(excel_data, field_mapping)
```

### 2. 验证代理 (ValidationAgent)

**位置**: `src/tools/validation_agent.py`

**主要功能**:
- 协调执行完整验证流程
- 生成AI改进建议
- 管理验证历史
- 提供验证摘要

**核心方法**:
```python
# 执行完整验证
validate_excel_data(excel_data, field_mapping)

# 验证LLM字段映射
validate_field_mapping_with_llm(excel_data, llm_response)

# 获取验证历史
get_validation_history()
```

### 3. 验证函数 (ValidationFunctions)

**位置**: `src/tools/validation_functions.py`

**主要功能**:
- 提供Function Calling接口
- 定义验证函数规范
- 执行验证函数
- 管理函数映射

**可用函数**:
- `validate_excel_structure`: 验证Excel结构
- `validate_field_mapping`: 验证字段映射
- `validate_data_content`: 验证数据内容
- `comprehensive_validation`: 综合验证
- `get_validation_summary`: 获取验证摘要

## API接口

### 1. Excel数据验证

**端点**: `POST /validate/excel`

**请求示例**:
```json
{
    "userInput": "验证数据质量",
    "excelFilename": "test.xlsx"
}
```

**响应示例**:
```json
{
    "code": 200,
    "message": "数据验证完成",
    "data": {
        "overall_valid": true,
        "validation_steps": [...],
        "summary": {...},
        "ai_recommendations": [...]
    }
}
```

### 2. Function Calling验证

**端点**: `POST /validate/function`

**请求示例**:
```json
{
    "userInput": "{\"function_name\":\"validate_excel_structure\",\"parameters\":{\"excel_filename\":\"test.xlsx\"}}",
    "excelFilename": "test.xlsx"
}
```

### 3. 获取验证函数列表

**端点**: `GET /validate/functions`

### 4. 获取验证历史

**端点**: `GET /validate/history`

## 使用方式

### 1. 通过聊天接口使用

在聊天中发送包含"验证"关键词的消息，系统会自动启用数据验证功能：

```
用户: "请验证这个Excel数据的质量"
系统: 自动执行数据验证并返回详细报告
```

### 2. 通过API直接调用

```python
import requests

# 验证Excel数据
response = requests.post("http://localhost:7860/validate/excel", json={
    "userInput": "验证数据",
    "excelFilename": "test.xlsx"
})

# 使用Function Calling
response = requests.post("http://localhost:7860/validate/function", json={
    "userInput": json.dumps({
        "function_name": "comprehensive_validation",
        "parameters": {"excel_filename": "test.xlsx"}
    }),
    "excelFilename": "test.xlsx"
})
```

### 3. 在代码中直接使用

```python
from src.tools.validation_agent import validation_agent
from src.tools.data_validator import data_validator

# 使用验证代理
result = validation_agent.validate_excel_data(excel_data)

# 使用验证器
structure_result = data_validator.validate_excel_data_structure(excel_data)
mapping_result = data_validator.validate_field_mapping(mapping, standard_fields)

# 或者使用文件名
from src.utils.excel_processor import excel_processor
excel_data = excel_processor.read_excel_to_json("test.xlsx")
structure_result = data_validator.validate_excel_data_structure(excel_data)
```

## 验证标准

### 1. 数据结构验证
- 检查必需字段是否存在
- 验证数据类型是否正确
- 检查数据一致性

### 2. 字段映射验证
- 检查映射完整性
- 验证映射唯一性
- 评估映射合理性

### 3. 数据内容验证
- 分析空值率
- 检查数据类型一致性
- 检测异常值
- 验证数据范围

### 4. 质量评分
- 数据完整性评分
- 数据一致性评分
- 数据准确性评分
- 映射合理性评分

## 配置说明

### 1. 验证规则配置

可以在 `DataValidator` 类中配置验证规则：

```python
self.validation_rules = {
    "required_fields": ["订单号", "运输日期"],
    "data_types": {"重量": "numeric"},
    "value_ranges": {"重量": {"min": 0, "max": 1000}},
    "format_patterns": {"订单号": r"^ORD\d+$"}
}
```

### 2. 质量评分权重

可以调整质量评分的权重：

```python
# 在 _calculate_quality_score 方法中调整
score -= len(validation_result["errors"]) * 20  # 错误扣分
score -= len(validation_result["warnings"]) * 5  # 警告扣分
```

## 测试

运行测试文件验证功能：

```bash
cd src/test
python validation_test.py
```

测试包括：
- 验证函数列表获取
- Excel数据验证
- Function Calling验证
- 字段映射验证
- 验证历史获取
- 聊天接口验证

## 扩展功能

### 1. 添加新的验证规则

在 `DataValidator` 类中添加新的验证方法：

```python
def validate_custom_rule(self, data, rule_config):
    """自定义验证规则"""
    # 实现验证逻辑
    pass
```

### 2. 添加新的验证函数

在 `validation_functions.py` 中添加新的函数定义：

```python
{
    "name": "validate_custom_function",
    "description": "自定义验证函数",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object"}
        },
        "required": ["data"]
    }
}
```

### 3. 集成新的验证提示词

在 `validation_prompts.py` 中添加新的提示词模板：

```python
custom_validation_prompt = ChatPromptTemplate.from_template("""
自定义验证提示词模板
{data}
""")
```

## 注意事项

1. **数据格式**: Excel数据必须符合指定的JSON格式
2. **字段映射**: 标准字段列表是固定的，不能随意修改
3. **性能考虑**: 大数据量验证可能需要较长时间
4. **错误处理**: 所有验证函数都包含完整的错误处理
5. **日志记录**: 验证过程会记录详细的日志信息

## 故障排除

### 常见问题

1. **验证失败**: 检查Excel数据格式是否正确
2. **函数调用失败**: 确认函数名称和参数格式
3. **性能问题**: 考虑分批处理大数据量
4. **内存不足**: 减少同时处理的数据量

### 调试方法

1. 查看日志文件了解详细错误信息
2. 使用测试文件验证功能
3. 检查API响应状态和错误信息
4. 验证数据格式和内容 