# 字段映射功能说明

## 概述

本项目新增了智能字段映射功能，能够自动将Excel表格中的个性化字段映射到标准化的运输管理系统字段。该功能基于LangGraph构建，包含多个处理节点和迭代逻辑，确保映射的准确性和完整性。

## 功能特性

### 🧩 核心节点设计

1. **字段预处理（Preprocess Fields）**
   - 清洗脏数据，去除空值率>90%的列
   - 提取字段特征：唯一性、重复率、数据类型
   - 过滤全0/全1的无意义列

2. **初步字段分类（Field Candidate Classification）**
   - 使用启发式规则进行字段分类
   - 支持日期类、编号类、数量类、名称类等分类
   - 结合字段名和数据类型特征

3. **标准字段匹配（Mapping to Standard Fields）**
   - 15个标准字段的智能匹配
   - 基于字段名相似度和数据特征的评分机制
   - 避免重复映射冲突

4. **冲突与缺失处理（Conflict & Missing Resolution）**
   - 自动检测和处理映射冲突
   - 优先级排序解决多对一映射
   - 缺失字段标记为"missing"

5. **映射验证（Mapping Validation）**
   - 业务逻辑校验
   - 关键字段存在性检查
   - 逻辑关系验证

6. **准确率评分（Confidence Scoring）**
   - 综合评分算法：字段覆盖率 + 关键字段正确性 + 验证通过率
   - 0-100分的准确率评估

### 🔄 迭代机制

- **冲突检测失败** → 回到分类阶段重新评估
- **验证不通过** → 回到映射阶段重新匹配
- **字段缺失推断** → 迭代尝试填补
- **最大迭代次数**：3次，避免无限循环

## API接口

### 字段映射接口

**端点**: `POST /field-mapping`

**请求参数**:
```json
{
    "userInput": "请帮我映射这些字段",
    "excelData": "JSON格式的Excel数据",
    "excelFilename": "Excel文件名（可选）",
    "stream": false
}
```

**响应格式**:
```json
{
    "success": true,
    "mapping": {
        "运输日期": "发货日期",
        "订单号": "订单编号",
        "运单号": "运单号码",
        "件数": "商品数量",
        "体积": "商品体积",
        "重量": "商品重量",
        "承运商": "承运商",
        "车型": "车型",
        "发货方名称": "发货方",
        "收货方名称": "收货方",
        "路顺": "路顺",
        "发货方编号": "missing",
        "收货方编码": "missing",
        "商品编码": "missing",
        "商品名称": "missing"
    },
    "confidence_score": 85.5,
    "validation": {
        "passed": true,
        "issues": []
    },
    "errors": [],
    "message": "字段映射完成"
}
```

## 标准字段列表

系统支持以下15个标准字段：

| 标准字段 | 说明 | 重要性 |
|---------|------|--------|
| 运输日期 | 货物发运日期 | 高 |
| 订单号 | 唯一订单标识 | 高 |
| 路顺 | 运输路线顺序 | 中 |
| 承运商 | 物流公司名称 | 中 |
| 运单号 | 运输单据号 | 高 |
| 车型 | 运输车辆类型 | 中 |
| 发货方编号 | 发货方编码 | 中 |
| 发货方名称 | 发货方名称 | 中 |
| 收货方编码 | 收货方编码 | 中 |
| 收货方名称 | 收货方名称 | 中 |
| 商品编码 | 商品唯一标识 | 中 |
| 商品名称 | 商品名称 | 中 |
| 件数 | 商品数量 | 中 |
| 体积 | 商品体积 | 中 |
| 重量 | 商品重量 | 中 |

## 使用方法

### 1. 启动服务

```bash
cd src
python main.py
```

### 2. 调用字段映射API

```python
import requests
import json

# 准备Excel数据
excel_data = {
    "订单编号": ["ORD001", "ORD002", "ORD003"],
    "发货日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "运单号码": ["WB001", "WB001", "WB002"],
    # ... 更多字段
}

# 发送请求
response = requests.post(
    "http://localhost:8000/field-mapping",
    json={
        "userInput": "请帮我映射这些字段",
        "excelData": json.dumps(excel_data, ensure_ascii=False),
        "stream": False
    }
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(f"映射准确率: {result['confidence_score']}%")
    print("字段映射结果:", result['mapping'])
```

### 3. 运行测试脚本

```bash
python test_field_mapping.py
```

## 技术架构

### 状态管理

使用 `FieldMappingState` 类管理整个映射过程的状态：

```python
class FieldMappingState(TypedDict):
    excel_data: Dict[str, Any]          # Excel原始数据
    preprocessed_fields: List[Dict]      # 预处理后的字段
    classified_fields: List[Dict]        # 分类后的字段
    initial_mapping: Dict[str, str]      # 初步映射结果
    final_mapping: Dict[str, str]        # 最终映射结果
    validation_results: Dict[str, Any]   # 验证结果
    confidence_score: float              # 准确率评分
    iteration_count: int                 # 迭代次数
    errors: List[str]                    # 错误信息
    messages: List                       # 消息历史
```

### 图结构

```
START → preprocess_fields → classify_fields → map_to_standard_fields 
         ↓
    resolve_conflicts → validate_mapping → calculate_confidence
         ↓
    [条件路由] → map_to_standard_fields (重试) 或 generate_output → END
```

## 配置说明

### 评分阈值

- **映射阈值**: 0.5（字段匹配的最低分数）
- **空值率阈值**: 0.9（超过90%空值的字段将被过滤）
- **唯一性阈值**: 0.8（编号类字段的最低唯一性要求）

### 优先级设置

冲突解决时的字段优先级（从高到低）：
1. 订单号
2. 运单号  
3. 运输日期
4. 件数/体积/重量

## 错误处理

系统包含完善的错误处理机制：

- **预处理错误**: 记录字段处理失败的原因
- **分类错误**: 记录字段分类失败的信息
- **映射错误**: 记录标准字段匹配失败的情况
- **验证错误**: 记录业务逻辑验证失败的问题

所有错误都会在最终响应中返回，便于调试和问题定位。

## 性能优化

- **数据预处理**: 过滤无效字段，减少后续处理量
- **智能分类**: 使用启发式规则快速分类，减少LLM调用
- **迭代控制**: 限制最大迭代次数，避免无限循环
- **状态缓存**: 使用LangGraph的状态管理，避免重复计算

## 扩展性

该架构支持以下扩展：

1. **新增标准字段**: 在 `map_to_standard_fields` 函数中添加新字段
2. **自定义验证规则**: 在 `validate_mapping` 函数中添加业务逻辑
3. **新的分类算法**: 在 `classify_fields` 函数中集成机器学习模型
4. **多语言支持**: 扩展字段名匹配的关键词库

## 注意事项

1. **数据格式**: Excel数据必须是JSON格式，支持列名:列数据的结构
2. **字段数量**: 建议单次处理不超过100个字段，避免性能问题
3. **迭代限制**: 系统最多迭代3次，超过后将强制输出结果
4. **错误容忍**: 系统会继续处理其他字段，即使部分字段处理失败

## 常见问题

### Q: 为什么某些字段被标记为"missing"？
A: 可能的原因：
- 字段名与标准字段差异太大
- 数据质量差（空值率高、全0/全1等）
- 字段类型不匹配（如日期字段包含非日期数据）

### Q: 如何提高映射准确率？
A: 建议：
- 使用更规范的字段命名
- 确保数据质量（减少空值、异常值）
- 提供更多样本数据
- 调整评分阈值参数

### Q: 支持哪些Excel格式？
A: 目前支持：
- JSON格式的列数据
- 通过excelFilename参数读取Excel文件
- 建议使用标准化的列名和数据格式 