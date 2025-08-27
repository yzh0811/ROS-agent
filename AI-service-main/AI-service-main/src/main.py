import re
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.configs.model_init import chat_model
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from src.schemas.model_response import Choice, ModelResponseData, ModelResponse
from src.prompts.prompts import user_prompt, greeting_card_prompt,test_prompt
from src.utils.excel_processor import excel_processor
from datetime import datetime
import threading
import sys
import io
import unicodedata

# 申明全局变量 全局调用
graph = None
field_mapping_graph = None  # 新增：字段映射图
# 获取当前时间并格式化为字符串（例如：2023-11-15 14:30:45）
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "AI-service-field-mapping"  # 项目名称

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
from src.configs.settings import manager, RegisterConfig
service_select = manager.get_service_config("greet-system")
app_host = service_select.app_host
app_port  = int(service_select.app_port)



class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    # messages: List[Message]
    # messages: str
    # user_input: str
    userInput: str
    stream: Optional[bool] = True
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    excelData: Optional[str] = None  # JSON格式的Excel数据
    excelFilename: Optional[str] = None  # Excel文件名



class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 新增：字段映射状态类
class FieldMappingState(TypedDict):
    excel_data: Dict[str, Any]  # Excel原始数据
    preprocessed_fields: List[Dict[str, Any]]  # 预处理后的字段
    classified_fields: List[Dict[str, Any]]  # 分类后的字段
    initial_mapping: Dict[str, str]  # 初步映射结果
    final_mapping: Dict[str, str]  # 最终映射结果
    validation_results: Dict[str, Any]  # 验证结果
    confidence_score: float  # 准确率评分
    iteration_count: int  # 迭代次数
    errors: List[str]  # 错误信息
    messages: Annotated[list, add_messages]  # 消息历史

# 新增：增强的LLM响应解析器
def parse_llm_response_robust(result_content: str) -> Dict[str, str]:
    """增强的LLM响应解析器，能够处理多种格式的响应"""
    try:
        import re
        
        # 标准字段列表
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量"
        ]
        
        # 尝试多种方式提取JSON
        json_patterns = [
            r'\{.*\}',  # 标准JSON
            r'```json\s*(\{.*?\})\s*```',  # Markdown代码块
            r'```\s*(\{.*?\})\s*```',  # 通用代码块
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, result_content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    llm_result = json.loads(json_str)
                    
                    # 处理不同的返回格式
                    if isinstance(llm_result, dict):
                        if "映射结果" in llm_result and isinstance(llm_result["映射结果"], dict):
                            # 格式1: {"映射结果": {...}}
                            return llm_result["映射结果"]
                        elif "mapping" in llm_result and isinstance(llm_result["mapping"], dict):
                            # 格式2: {"mapping": {...}}
                            return llm_result["mapping"]
                        elif any(k in llm_result for k in standard_fields):
                            # 格式3: 直接字段映射 {"运输日期": "字段名", ...}
                            return {k: llm_result.get(k, "missing") for k in standard_fields}
                        elif "confidence" in llm_result or "准确率" in llm_result:
                            # 格式4: 包含置信度的映射
                            # 尝试从响应中提取字段映射信息
                            return extract_mapping_from_text(result_content, standard_fields)
                    
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # 如果所有JSON解析都失败，尝试从文本中提取
        return extract_mapping_from_text(result_content, standard_fields)
        
    except Exception as e:
        logger.error(f"LLM响应解析失败: {str(e)}")
        return {}
    
def apply_similarity_fill(initial_mapping: dict, preprocessed_fields: list, threshold: float = 0.4, only_missing: bool = True) -> dict:
    import re, unicodedata
    def _norm(s: str) -> str:
        if s is None: return ""
        t = unicodedata.normalize("NFKC", str(s)).lower()
        t = re.sub(r"[_\-\./\\]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    def _tok(s: str):
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fa5]+", _norm(s))
    def _edit_sim(a: str, b: str) -> float:
        a, b = _norm(a), _norm(b)
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        la, lb = len(a), len(b)
        dp = [[0]*(lb+1) for _ in range(la+1)]
        for i in range(la+1): dp[i][0] = i
        for j in range(lb+1): dp[0][j] = j
        for i in range(1, la+1):
            for j in range(1, lb+1):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]!=b[j-1]))
        dist = dp[la][lb]
        return 1 - dist / max(1, max(la, lb))
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(_tok(a)), set(_tok(b))
        return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0
    UNIT_HINTS = {"体积": ["cbm","m3","m³","立方","容积","cubic","vol"], "重量": ["kg","公斤","千克","吨","lb","wt","weight"], "件数": ["pcs","box","qty","count","箱","包","数量","件"]}
    TOTAL_HINTS = ["总","合计","总计","total","sum"]
    def _bonus(name: str, field: str) -> float:
        n = _norm(name); b = 0.0
        if field in UNIT_HINTS and any(h in n for h in UNIT_HINTS[field]): b += 0.2
        if field in ["件数","体积","重量"] and any(h in n for h in TOTAL_HINTS): b += 0.2
        return min(b, 0.4)

    std_fields = ["运输日期","订单号","路顺","承运商","运单号","车型","发货方编号","发货方名称","收货方编码","收货方名称","商品编码","商品名称","件数","体积","重量"]
    if not preprocessed_fields: return initial_mapping
    cols = [f.get("name") for f in preprocessed_fields if f.get("name")]
    used = set(v for v in initial_mapping.values() if v and v != "missing")
    updated = dict(initial_mapping)

    for f in std_fields:
        if only_missing and updated.get(f) and updated[f] != "missing":
            continue
        best_name, best_score = None, 0.0
        for name in cols:
            if name in used: continue
            score = 0.6*_edit_sim(name, f) + 0.4*_jaccard(name, f) + _bonus(name, f)
            if score > best_score:
                best_score, best_name = score, name
        if best_name and best_score >= threshold:
            updated[f] = best_name
            used.add(best_name)
    return updated

def extract_mapping_from_text(text: str, standard_fields: List[str]) -> Dict[str, str]:
    """从文本中提取字段映射信息"""
    try:
        mapping = {field: "missing" for field in standard_fields}
        
        # 尝试从文本中识别字段映射
        for field in standard_fields:
            # 查找包含字段名的行
            field_pattern = rf'{field}[：:]\s*["""]?([^"""\n]+)["""]?'
            match = re.search(field_pattern, text)
            if match:
                mapped_field = match.group(1).strip()
                if mapped_field and mapped_field != "missing":
                    mapping[field] = mapped_field
        
        return mapping
        
    except Exception as e:
        logger.error(f"文本提取映射失败: {str(e)}")
        return {field: "missing" for field in standard_fields}

def try_hybrid_mapping(state: FieldMappingState, llm_response: str) -> Dict[str, str]:
    """混合策略：结合LLM响应和规则映射"""
    try:
        # 标准字段列表
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量"
        ]
        
        # 初始化映射
        hybrid_mapping = {field: "missing" for field in standard_fields}
        
        # 尝试从LLM响应中提取部分信息
        partial_mapping = extract_mapping_from_text(llm_response, standard_fields)
        
        # 获取规则映射作为补充
        rule_mapping = get_rule_based_mapping(state)
        
        # 合并映射：优先使用LLM结果，规则映射作为补充
        for field in standard_fields:
            if partial_mapping.get(field) != "missing":
                hybrid_mapping[field] = partial_mapping[field]
            elif rule_mapping.get(field) != "missing":
                hybrid_mapping[field] = rule_mapping[field]
        
        logger.info(f"🔍 [混合策略] 成功映射 {len([v for v in hybrid_mapping.values() if v != 'missing'])} 个字段")
        return hybrid_mapping
        
    except Exception as e:
        logger.error(f"混合策略失败: {str(e)}")
        return {field: "missing" for field in standard_fields}

def get_rule_based_mapping(state: FieldMappingState) -> Dict[str, str]:
    """获取基于规则的映射"""
    try:
        classified_fields = state.get("classified_fields", [])
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量"
        ]
        
        rule_mapping = {}
        used_fields = set()
        
        for standard_field in standard_fields:
            best_match = "missing"
            best_score = 0
            
            for field in classified_fields:
                if field["name"] in used_fields:
                    continue
                    
                score = calculate_mapping_score(standard_field, field)
                if score > best_score:
                    best_score = score
                    best_match = field["name"]
            
            if best_score > 0.3:
                rule_mapping[standard_field] = best_match
                used_fields.add(best_match)
            else:
                rule_mapping[standard_field] = "missing"
        
        return rule_mapping
        
    except Exception as e:
        logger.error(f"规则映射获取失败: {str(e)}")
        return {field: "missing" for field in standard_fields}

# 新增：字段预处理节点
def preprocess_fields(state: FieldMappingState) -> FieldMappingState:
    """字段预处理：清洗脏数据，提取字段特征"""
    try:
        # 添加LangSmith追踪标签
        from langsmith import traceable
        import uuid
        
        # 创建追踪ID
        trace_id = str(uuid.uuid4())
        
        excel_data = state["excel_data"]
        preprocessed_fields = []
        
        # 确保excel_data是字典格式，如果是列表则转换为字典
        if isinstance(excel_data, list) and len(excel_data) > 0:
            # Excel处理器返回的是行记录列表，需要转换为列数据格式
            if isinstance(excel_data[0], dict):
                # 转换为列名:列数据的格式
                columns = list(excel_data[0].keys())
                converted_data = {}
                for col in columns:
                    converted_data[col] = [row.get(col) for row in excel_data]
                excel_data = converted_data
        
        for column_name, column_data in excel_data.items():
            if not isinstance(column_data, list):
                continue
                
            # 计算字段特征
            total_rows = len(column_data)
            non_null_count = sum(1 for value in column_data if value is not None and value != "")
            null_rate = (total_rows - non_null_count) / total_rows if total_rows > 0 else 1.0
            
            # 发货方相关字段的白名单保护（即使空值率高也要保留）
            shipper_keywords = [
                "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",
                "站名", "网名", "店名", "仓名", "厂名",
                "站号", "网号", "店号", "仓号", "厂号",
                "站编码", "网编码", "店编码", "仓编码", "厂编码",
                "发货", "发送", "始发", "起运", "发运"
            ]
            
            is_shipper_field = any(keyword in column_name for keyword in shipper_keywords)
            
            # 过滤条件：空值率>90%、全0、全1的列（发货方字段除外）
            if null_rate > 0.9 and not is_shipper_field:
                if is_shipper_field:
                    logger.info(f"🔍 [预处理] 保护发货方字段: {column_name} (空值率: {null_rate:.2%})")
                continue
            elif is_shipper_field and null_rate > 0.9:
                logger.info(f"🔍 [预处理] 发货方字段被保护: {column_name} (空值率: {null_rate:.2%})")
                
            # 计算唯一性
            unique_values = set(str(v) for v in column_data if v is not None and v != "")
            uniqueness = len(unique_values) / total_rows if total_rows > 0 else 0
            
            # 检测数据类型
            data_type = "text"
            if all(isinstance(v, (int, float)) for v in column_data if v is not None):
                data_type = "numeric"
            elif all(isinstance(v, str) and len(v) == 10 for v in column_data if v is not None):
                # 简单日期检测
                data_type = "date"
            
            field_info = {
                "name": column_name,
                "data": column_data,
                "null_rate": null_rate,
                "uniqueness": uniqueness,
                "data_type": data_type,
                "total_rows": total_rows,
                "non_null_count": non_null_count
            }
            preprocessed_fields.append(field_info)
        
        state["preprocessed_fields"] = preprocessed_fields
        state["messages"].append({"role": "system", "content": f"字段预处理完成，共处理 {len(preprocessed_fields)} 个有效字段"})
        
        # 记录LangSmith追踪信息
        logger.info(f"🔍 [LangSmith] 节点: preprocess_fields | 追踪ID: {trace_id} | 处理字段数: {len(preprocessed_fields)}")
        
        return state
        
    except Exception as e:
        state["errors"].append(f"字段预处理失败: {str(e)}")
        return state

# 新增：字段分类节点
def classify_fields(state: FieldMappingState) -> FieldMappingState:
    """初步字段分类：将字段分配到候选类别"""
    try:
        # 添加LangSmith追踪标签
        from langsmith import traceable
        import uuid
        
        # 创建追踪ID
        trace_id = str(uuid.uuid4())
        
        preprocessed_fields = state["preprocessed_fields"]
        classified_fields = []
        
        for field in preprocessed_fields:
            field_name = field["name"].lower()
            data_type = field["data_type"]
            uniqueness = field["uniqueness"]
            
            # 启发式规则分类
            category = "unknown"
            
            # 日期类检测
            if any(keyword in field_name for keyword in ["时间", "日期", "date", "time"]) or data_type == "date":
                category = "date"
            # 编号类检测
            elif any(keyword in field_name for keyword in ["编号", "编码", "code", "id", "号"]) and uniqueness > 0.8:
                category = "identifier"
            # 数量类检测
            elif any(keyword in field_name for keyword in ["数量", "件数", "体积", "重量", "count", "volume", "weight"]) and data_type == "numeric":
                category = "quantity"
            # 名称类检测
            elif any(keyword in field_name for keyword in ["名称", "name", "地址", "address", "公司", "company"]):
                category = "name"
            
            field["category"] = category
            classified_fields.append(field)
        
        state["classified_fields"] = classified_fields
        state["messages"].append({"role": "system", "content": f"字段分类完成，共分类 {len(classified_fields)} 个字段"})
        
        # 记录LangSmith追踪信息
        logger.info(f"🏷️ [LangSmith] 节点: classify_fields | 追踪ID: {trace_id} | 分类字段数: {len(classified_fields)}")
        
        return state
        
    except Exception as e:
        state["errors"].append(f"字段分类失败: {str(e)}")
        return state

# 新增：LLM智能字段匹配节点
def llm_map_to_standard_fields(state: FieldMappingState) -> FieldMappingState:
    """使用LLM进行智能字段匹配"""
    try:
        from src.prompts.prompts import test_prompt
        from src.configs.model_init import chat_model
        
        # 添加LangSmith追踪标签
        from langsmith import traceable
        import uuid
        
        # 创建追踪ID
        trace_id = str(uuid.uuid4())
        logger.info(f"🧠 [LangSmith] 开始LLM智能字段映射 | 追踪ID: {trace_id}")
        
        # 准备Excel数据供LLM分析
        excel_data = state["excel_data"]
        
        # 添加调试日志 - 显示LLM收到的字段信息
        logger.info(f"🔍 [LLM] excel_data类型: {type(excel_data)}")
        logger.info(f"🔍 [LLM] excel_data长度: {len(excel_data) if hasattr(excel_data, '__len__') else 'N/A'}")
        
        if isinstance(excel_data, dict):
            field_names = list(excel_data.keys())
            logger.info(f"🔍 [LLM] 准备发送给LLM的字段: {field_names}")
            logger.info(f"🔍 [LLM] 字段数量: {len(field_names)}")
            
            # 检查是否有发货方相关的字段
            shipper_keywords = [
                "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",
                "站名", "网名", "店名", "仓名", "厂名",
                "站号", "网号", "店号", "仓号", "厂号",
                "站编码", "网编码", "店编码", "仓编码", "厂编码",
                "发货", "发送", "始发", "起运", "发运"
            ]
            
            shipper_fields = []
            for field in field_names:
                if any(keyword in field for keyword in shipper_keywords):
                    shipper_fields.append(field)
            
            if shipper_fields:
                logger.info(f"🔍 [LLM] 发现发货方相关字段: {shipper_fields}")
            else:
                logger.warning(f"⚠️ [LLM] 未发现发货方相关字段！")
                logger.info(f"🔍 [LLM] 所有字段: {field_names}")
        elif isinstance(excel_data, list):
            logger.info(f"🔍 [LLM] excel_data是列表类型，长度: {len(excel_data)}")
            if len(excel_data) > 0:
                logger.info(f"🔍 [LLM] 第一行数据类型: {type(excel_data[0])}")
                if isinstance(excel_data[0], dict):
                    field_names = list(excel_data[0].keys())
                    logger.info(f"🔍 [LLM] 列表中的字段: {field_names}")
                    
                    # 检查是否有发货方相关的字段
                    shipper_keywords = [
                        "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",
                        "站名", "网名", "店名", "仓名", "厂名",
                        "站号", "网号", "店号", "仓号", "厂号",
                        "站编码", "网编码", "店编码", "仓编码", "厂编码",
                        "发货", "发送", "始发", "起运", "发运"
                    ]
                    
                    shipper_fields = []
                    for field in field_names:
                        if any(keyword in field for keyword in shipper_keywords):
                            shipper_fields.append(field)
                    
                    if shipper_fields:
                        logger.info(f"🔍 [LLM] 发现发货方相关字段: {shipper_fields}")
                    else:
                        logger.warning(f"⚠️ [LLM] 未发现发货方相关字段！")
                        logger.info(f"🔍 [LLM] 所有字段: {field_names}")
        else:
            logger.warning(f"⚠️ [LLM] excel_data是未知类型: {type(excel_data)}")
        
        # 转换为JSON字符串
        excel_json_str = json.dumps(excel_data, ensure_ascii=False)
        logger.info(f"📊 准备发送给LLM的数据长度: {len(excel_json_str)} 字符")
        
        # 使用test_prompt进行LLM分析
        try:
            system_template = test_prompt.messages[0].prompt.template
            formatted_system_content = system_template.format(excel_data=excel_json_str)
            
            prompt = [{"role": "system", "content": formatted_system_content}]
            
            # 调用LLM
            response = chat_model.invoke(prompt)
            result_content = response.content
            
            # 添加调试日志 - 显示LLM的原始响应
            logger.info(f"🔍 [LLM] LLM原始响应长度: {len(result_content)} 字符")
            logger.info(f"🔍 [LLM] LLM响应前100字符: {result_content[:100]}...")
            
            # 增强的LLM响应解析器
            try:
                initial_mapping = parse_llm_response_robust(result_content)
                
                if initial_mapping:
                    # LLM解析成功，使用LLM结果
                    state["initial_mapping"] = initial_mapping
                    state["confidence_score"] = 85.0  # 默认置信度
                    state["messages"].append({
                        "role": "assistant",
                        "content": f"LLM字段映射完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段，置信度: 85%"
                    })
                    
                    # 相似度精排补全（仅补全missing，不覆盖LLM已有映射）
                    try:
                        pre_fields = state.get("preprocessed_fields", [])
                        sim_updated = apply_similarity_fill(state["initial_mapping"], pre_fields, threshold=0.4, only_missing=True)
                        if sim_updated != state["initial_mapping"]:
                            filled_cnt = len([1 for k in sim_updated if sim_updated[k] != "missing" and state["initial_mapping"].get(k, "missing") == "missing"]) 
                            logger.info(f"🔍 [相似度补全] 在LLM结果基础上补全 {filled_cnt} 个字段")
                            state["initial_mapping"] = sim_updated
                    except Exception as _e:
                        logger.warning(f"⚠️ [相似度补全] LLM后补全失败: {_e}")
                else:
                    # LLM解析失败，尝试混合策略
                    logger.warning("⚠️ LLM响应解析失败，尝试混合策略")
                    initial_mapping = try_hybrid_mapping(state, result_content)
                    state["initial_mapping"] = initial_mapping
                    state["messages"].append({
                        "role": "system",
                        "content": f"混合策略完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段"
                    })
                    
            except Exception as parse_error:
                logger.error(f"LLM响应处理失败: {str(parse_error)}")
                state["errors"].append(f"LLM响应处理失败: {str(parse_error)}")
                # 最后才降级到规则匹配
                return rule_based_mapping_fallback(state)
                
        except Exception as llm_error:
            state["errors"].append(f"LLM调用失败: {str(llm_error)}")
            # 降级到规则匹配
            return rule_based_mapping_fallback(state)
        
        return state
        
    except Exception as e:
        state["errors"].append(f"LLM字段匹配失败: {str(e)}")
        # 降级到规则匹配
        return rule_based_mapping_fallback(state)

# 新增：规则匹配降级方案
def rule_based_mapping_fallback(state: FieldMappingState) -> FieldMappingState:
    """规则匹配降级方案 - 增强版"""
    try:
        logger.info("🔍 [规则降级] 开始规则匹配降级")
        
        # 使用增强的规则映射函数
        initial_mapping = get_rule_based_mapping(state)
        
        # 智能补充映射：使用配置的映射规则进行补充
        try:
            from src.configs.manual_mapping_rules import get_mapping_rules
            
            mapping_rules = get_mapping_rules()
            preprocessed_fields = state.get("preprocessed_fields", [])
            
            # 获取所有字段名称
            available_fields = [field["name"] for field in preprocessed_fields]
            logger.info(f"🔍 [规则回退] 可用字段: {available_fields}")
            
            # 应用映射规则
            for standard_field, keywords in mapping_rules.items():
                if initial_mapping.get(standard_field) == "missing":
                    for field_name in available_fields:
                        if any(keyword in field_name for keyword in keywords):
                            initial_mapping[standard_field] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充 {standard_field}: {field_name}")
                            break
                            
        except ImportError:
            logger.warning("⚠️ [规则回退] 无法导入手动映射规则，使用内置规则")
            # 使用内置规则作为备选
            if initial_mapping.get("发货方编号") == "missing" or initial_mapping.get("发货方名称") == "missing":
                logger.info("🔍 [规则回退] 尝试智能补充发货方映射...")
                
                # 发货方相关关键词
                shipper_id_keywords = ["站点编号", "网点编号", "门店编号", "仓库编号", "配送中心编号", "DC编号", "工厂编号", "站号", "网号", "店号", "仓号", "厂号"]
                shipper_name_keywords = ["站点名称", "网点名称", "门店名称", "仓库名称", "配送中心名称", "DC名称", "工厂名称", "站名", "网名", "店名", "仓名", "厂名"]
                
                # 尝试匹配发货方编号
                if initial_mapping.get("发货方编号") == "missing":
                    for field in preprocessed_fields:
                        field_name = field["name"]
                        if any(keyword in field_name for keyword in shipper_id_keywords):
                            initial_mapping["发货方编号"] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充发货方编号: {field_name}")
                            break
                
                # 尝试匹配发货方名称
                if initial_mapping.get("发货方名称") == "missing":
                    for field in preprocessed_fields:
                        field_name = field["name"]
                        if any(keyword in field_name for keyword in shipper_name_keywords):
                            initial_mapping["发货方名称"] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充发货方名称: {field_name}")
                            break
        
        # 相似度精排补全（仅补全missing）
        try:
            pre_fields = state.get("preprocessed_fields", [])
            sim_updated = apply_similarity_fill(initial_mapping, pre_fields, threshold=0.4, only_missing=True)
            if sim_updated != initial_mapping:
                filled_cnt = len([1 for k in sim_updated if sim_updated[k] != "missing" and initial_mapping.get(k, "missing") == "missing"]) 
                logger.info(f"🔍 [相似度补全] 在规则降级结果基础上补全 {filled_cnt} 个字段")
                initial_mapping = sim_updated
        except Exception as _e:
            logger.warning(f"⚠️ [相似度补全] 规则降级后补全失败: {_e}")
        
        # # 确保所有标准字段都有映射
        # for field in standard_fields:
        #     if field not in initial_mapping:
        #         initial_mapping[field] = "missing"
        
        state["initial_mapping"] = initial_mapping
        state["messages"].append({"role": "system", "content": f"规则匹配降级完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段"})
        
        return state
        
    except Exception as e:
        state["errors"].append(f"规则匹配降级失败: {str(e)}")
        return state

# 新增：映射评分函数
def calculate_mapping_score(standard_field: str, field: Dict[str, Any]) -> float:
    """计算字段映射的匹配分数"""
    score = 0.0
    field_name = field["name"].lower()
    
    # 基于字段名的匹配
    if standard_field == "订单号":
        if any(keyword in field_name for keyword in ["订单", "单号", "order", "订单号"]):
            score += 0.6
        if field["uniqueness"] > 0.9:
            score += 0.4
    elif standard_field == "运单号":
        if any(keyword in field_name for keyword in ["运单", "运单号", "waybill", "运单号"]):
            score += 0.6
        if field["uniqueness"] > 0.7:
            score += 0.3
    elif standard_field == "运输日期":
        if any(keyword in field_name for keyword in ["运输", "发运", "日期", "时间", "date", "time"]):
            score += 0.6
        if field["data_type"] == "date":
            score += 0.4
    elif standard_field == "件数":
        if any(keyword in field_name for keyword in ["件数", "数量", "count", "qty", "件", "总件数", "件数合计", "件数总计", "箱数", "总箱数", "包数", "总包数", "pcs", "box"]):
            score += 0.8
        if field["data_type"] == "numeric":
            score += 0.2
        # 额外加分：如果字段名包含"总"、"合计"、"总计"等
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "体积":
        if any(keyword in field_name for keyword in ["体积", "volume", "立方", "m3", "总体积", "体积合计", "体积总计", "容积", "空间", "体积量", "vol", "cbm", "cubic"]):
            score += 0.8
        if field["data_type"] == "numeric":
            score += 0.2
        # 额外加分：如果字段名包含"总"、"合计"、"总计"等
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "重量":
        if any(keyword in field_name for keyword in ["重量", "weight", "公斤", "kg", "重", "总重量", "重量合计", "重量总计", "千克", "吨", "斤", "磅", "wt", "ton", "lb"]):
            score += 0.8
        if field["data_type"] == "numeric":
            score += 0.2
        # 额外加分：如果字段名包含"总"、"合计"、"总计"等
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "承运商":
        # 扩展承运商的关键词覆盖
        if any(keyword in field_name for keyword in [
            "承运商", "物流", "运输", "carrier", "logistics",
            "运输公司", "物流公司", "快递公司", "配送公司",  # 新增：公司类型
            "供应商", "分包商", "指定承运商", "合作承运商"  # 新增：供应商相关
        ]):
            score += 0.7
        if field["category"] == "name":
            score += 0.3
    elif standard_field == "车型":
        if any(keyword in field_name for keyword in ["车型", "车辆", "vehicle", "truck", "车"]):
            score += 0.8
        if field["category"] == "name":
            score += 0.2
    elif standard_field == "发货方名称":
        # 扩展发货方名称的关键词覆盖
        if any(keyword in field_name for keyword in [
            "发货", "发送", "sender", "shipper", "发",
            "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",  # 新增：站点/网点/门店等
            "站名", "网名", "店名", "仓名", "厂名",  # 新增：站点/网点/门店等的名称
            "发货点", "始发点", "起运点", "发运点"  # 新增：发货点相关
        ]):
            score += 0.7
        if field["category"] == "name":
            score += 0.3
    elif standard_field == "收货方名称":
        if any(keyword in field_name for keyword in ["收货", "接收", "receiver", "consignee", "收"]):
            score += 0.7
        if field["category"] == "name":
            score += 0.3
    elif standard_field == "发货方编号":
        # 扩展发货方编号的关键词覆盖
        if any(keyword in field_name for keyword in [
            "发货", "发送", "sender", "shipper", "发"
        ]) and any(keyword in field_name for keyword in [
            "编号", "编码", "code", "id"
        ]):
            score += 0.8
        # 新增：站点/网点/门店/仓库/工厂编号的识别（不需要同时包含"发货"关键词）
        elif any(keyword in field_name for keyword in [
            "站点编号", "网点编号", "门店编号", "仓库编号", "配送中心编号", "DC编号", "工厂编号",
            "站号", "网号", "店号", "仓号", "厂号", "配送号"
        ]):
            score += 0.9  # 给这些明确的发货方编号更高分数
        # 新增：站点/网点/门店/仓库/工厂的编码识别
        elif any(keyword in field_name for keyword in [
            "站点编码", "网点编码", "门店编码", "仓库编码", "配送中心编码", "DC编码", "工厂编码",
            "站编码", "网编码", "店编码", "仓编码", "厂编码"
        ]):
            score += 0.9  # 给这些明确的发货方编码更高分数
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "收货方编码":
        if any(keyword in field_name for keyword in ["收货", "接收", "receiver"]) and any(keyword in field_name for keyword in ["编号", "编码", "code", "id"]):
            score += 0.8
        # 新增：直接识别送货点、配送点等收货相关字段
        elif any(keyword in field_name for keyword in ["送货点", "送货点编号", "配送点", "配送点编号", "目的地", "目的地编号", "客户", "客户编号", "客户号", "终端", "终端编号"]):
            score += 0.9  # 给这些明确的收货方字段更高分数
        # 额外加分：如果字段名包含收货相关词汇
        if any(keyword in field_name.lower() for keyword in ["送货", "收货", "配送", "目的地", "客户", "终端"]):
            score += 0.3
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "收货方名称":
        if any(keyword in field_name for keyword in ["收货", "接收", "receiver"]) and any(keyword in field_name for keyword in ["名称", "name", "收货方名称"]):
            score += 0.8
        # 新增：直接识别送货点、配送点等收货相关字段
        elif any(keyword in field_name for keyword in ["送货点", "送货点名称", "配送点", "配送点名称", "目的地", "目的地名称", "客户", "客户名称", "客户名", "终端", "终端名称"]):
            score += 0.9  # 给这些明确的收货方字段更高分数
        # 额外加分：如果字段名包含收货相关词汇
        if any(keyword in field_name.lower() for keyword in ["送货", "收货", "配送", "目的地", "客户", "终端"]):
            score += 0.3
        if field["category"] == "name":
            score += 0.2
    elif standard_field == "商品编码":
        # 扩展商品编码的关键词覆盖
        if any(keyword in field_name for keyword in [
            "商品", "产品", "product", "goods", "货物", "货品"
        ]) and any(keyword in field_name for keyword in [
            "编号", "编码", "code", "id", "货号", "品号"
        ]):
            score += 0.8
        # 新增：直接的商品编码识别（不需要同时包含"商品"关键词）
        elif any(keyword in field_name for keyword in [
            "货号", "品号", "SKU", "sku", "商品代码", "产品代码", "item id", "item id", "item code", "product code", "product id", "goods code", "material code"
        ]):
            score += 0.9  # 给这些明确的商品编码更高分数
        # 额外加分：如果字段名包含英文商品相关词汇
        if any(keyword in field_name.lower() for keyword in ["item", "product", "goods", "material", "sku"]):
            score += 0.3
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "商品名称":
        # 扩展商品名称的关键词覆盖
        if any(keyword in field_name for keyword in [
            "商品", "产品", "product", "goods", "货物", "货品"
        ]) and any(keyword in field_name for keyword in [
            "名称", "name", "品名", "货名"
        ]):
            score += 0.8
        # 新增：直接的商品名称识别（不需要同时包含"商品"关键词）
        elif any(keyword in field_name for keyword in [
            "品名", "货名", "商品描述", "产品描述", "货物描述", "item description", "item desc", "product name", "product description", "goods name", "material description"
        ]):
            score += 0.9  # 给这些明确的商品名称更高分数
        # 额外加分：如果字段名包含英文商品相关词汇
        if any(keyword in field_name.lower() for keyword in ["item", "product", "goods", "material", "description", "desc", "name"]):
            score += 0.3
        if field["category"] == "name":
            score += 0.2
    elif standard_field == "路顺":
        if any(keyword in field_name for keyword in ["路顺", "顺序", "sequence", "order", "路线"]):
            score += 0.9
        if field["data_type"] == "numeric":
            score += 0.1
    
    return score

# 新增：冲突与缺失处理节点
def resolve_conflicts_and_missing(state: FieldMappingState) -> FieldMappingState:
    """冲突与缺失处理：检查冲突并修正映射"""
    try:
        initial_mapping = state.get("initial_mapping", {})
        
        # 添加调试日志 - 冲突处理前的映射状态
        logger.info(f"🔍 [冲突处理] 开始处理冲突与缺失")
        logger.info(f"🔍 [冲突处理] 冲突处理前的映射: {json.dumps(initial_mapping, ensure_ascii=False)}")
        
        # 如果initial_mapping为空，创建一个默认映射
        if not initial_mapping:
            logger.warning("⚠️ initial_mapping为空，创建默认映射")
            standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型", 
                "发货方编号", "发货方名称", "收货方编码", "收货方名称", 
                "商品编码", "商品名称", "件数", "体积", "重量"
            ]
            initial_mapping = {field: "missing" for field in standard_fields}
        
        final_mapping = initial_mapping.copy()
        
        # 检查是否有多个个性化字段映射到同一个标准化字段
        value_counts = {}
        for standard_field, personalized_field in initial_mapping.items():
            if personalized_field != "missing":
                if personalized_field in value_counts:
                    value_counts[personalized_field].append(standard_field)
                else:
                    value_counts[personalized_field] = [standard_field]
        
        # 添加调试日志 - 检测到的冲突
        if value_counts:
            logger.info(f"🔍 [冲突处理] 检测到的字段映射关系:")
            for personalized_field, standard_fields in value_counts.items():
                if len(standard_fields) > 1:
                    logger.info(f"🔍 [冲突处理] 冲突: {personalized_field} -> {standard_fields}")
                else:
                    logger.info(f"🔍 [冲突处理] 正常: {personalized_field} -> {standard_fields}")
        else:
            logger.info(f"🔍 [冲突处理] 没有检测到任何字段映射关系")
        
        # 处理冲突
        conflict_count = 0
        for personalized_field, standard_fields in value_counts.items():
            if len(standard_fields) > 1:
                conflict_count += 1
                logger.info(f"🔍 [冲突处理] 处理第{conflict_count}个冲突: {personalized_field} -> {standard_fields}")
                
                # 保留优先级最高的映射，其他设为missing
                priority_order = ["件数", "体积", "重量", "商品编码", "商品名称", "收货方编码", "收货方名称", "发货方编号", "发货方名称", "运输日期", "订单号","运单号"]
                
                # 计算每个字段的优先级分数
                field_scores = {}
                for field in standard_fields:
                    if field in priority_order:
                        score = priority_order.index(field)
                        field_scores[field] = score
                        logger.info(f"🔍 [冲突处理] {field} 优先级分数: {score}")
                    else:
                        field_scores[field] = 999
                        logger.info(f"🔍 [冲突处理] {field} 优先级分数: 999 (不在优先级列表中)")
                
                best_field = max(standard_fields, key=lambda x: priority_order.index(x) if x in priority_order else 999)
                logger.info(f"🔍 [冲突处理] 选择保留: {best_field} (优先级分数: {field_scores[best_field]})")
                
                for field in standard_fields:
                    if field != best_field:
                        final_mapping[field] = "missing"
                        logger.info(f"🔍 [冲突处理] 设为missing: {field} (优先级分数: {field_scores[field]})")
        
        if conflict_count == 0:
            logger.info(f"🔍 [冲突处理] 没有检测到任何冲突")
        
        # 添加调试日志 - 冲突处理后的映射状态
        logger.info(f"🔍 [冲突处理] 冲突处理后的映射: {json.dumps(final_mapping, ensure_ascii=False)}")
        
        # 检查发货方相关字段的状态变化
        shipper_fields = ["发货方名称", "发货方编号"]
        for field in shipper_fields:
            if initial_mapping.get(field) != final_mapping.get(field):
                logger.warning(f"⚠️ [冲突处理] {field} 状态发生变化: {initial_mapping.get(field)} -> {final_mapping.get(field)}")
        
        state["final_mapping"] = final_mapping
        state["messages"].append({"role": "system", "content": "冲突与缺失处理完成"})
        return state
        
    except Exception as e:
        state["errors"].append(f"冲突与缺失处理失败: {str(e)}")
        return state

# 新增：映射验证节点
def validate_mapping(state: FieldMappingState) -> FieldMappingState:
    """映射验证：业务逻辑校验"""
    try:
        final_mapping = state["final_mapping"]
        validation_results = {"passed": True, "issues": []}
        
        # 放宽验证条件 - 不再强制要求任何字段
        important_fields = ["订单号", "运单号", "运输日期"]  # 重要字段，仅提醒
        
        # 检查重要字段（仅提醒，不影响验证通过）
        missing_important = [field for field in important_fields if final_mapping.get(field) == "missing"]
        if missing_important:
            validation_results["issues"].append(f"建议补充重要字段: {', '.join(missing_important)}")
        
        # 逻辑关系验证
        if final_mapping.get("路顺") != "missing" and final_mapping.get("运单号") == "missing":
            validation_results["issues"].append("路顺字段存在但缺少运单号")
        
        # 数量字段验证
        quantity_fields = ["件数", "体积", "重量"]
        if all(final_mapping.get(field) == "missing" for field in quantity_fields):
            validation_results["issues"].append("缺少所有数量相关字段")
        
        # 设置验证通过 - 不再强制要求关键字段
        validation_results["passed"] = True
        
        state["validation_results"] = validation_results
        state["messages"].append({"role": "system", "content": f"映射验证完成，通过: {validation_results['passed']}"})
        return state
        
    except Exception as e:
        state["errors"].append(f"映射验证失败: {str(e)}")
        return state

# 新增：准确率评分节点
def calculate_confidence_score(state: FieldMappingState) -> FieldMappingState:
    """计算准确率评分"""
    try:
        final_mapping = state["final_mapping"]
        validation_results = state["validation_results"]
        
        # 字段覆盖率
        total_fields = len(final_mapping)
        mapped_fields = len([v for v in final_mapping.values() if v != "missing"])
        coverage_rate = mapped_fields / total_fields if total_fields > 0 else 0
        
        # 字段重要性分层评分 - 调整权重分配
        important_fields = ["订单号", "运单号", "运输日期", "件数", "体积", "重量"]  # 重要字段
        other_fields = ["路顺", "承运商", "车型", "发货方名称", "收货方名称", "商品名称"]  # 一般重要
        
        important_score = sum(1 for field in important_fields if final_mapping.get(field) != "missing") / len(important_fields)
        other_score = sum(1 for field in other_fields if final_mapping.get(field) != "missing") / len(other_fields)
        
        # 验证通过率 - 现在总是1.0，因为不再强制要求关键字段
        validation_score = 1.0
        
        # 综合评分 - 重新分配权重
        confidence_score = (0.6 * important_score +    # 重要字段权重60%
                          0.3 * other_score +         # 其他字段权重30% 
                          0.1 * validation_score)     # 验证通过权重10%
        confidence_score = round(confidence_score * 100, 1)
        
        state["confidence_score"] = confidence_score
        state["messages"].append({"role": "system", "content": f"准确率评分完成: {confidence_score}%"})
        return state
        
    except Exception as e:
        state["errors"].append(f"准确率评分失败: {str(e)}")
        return state

# 新增：结果生成节点
def generate_output(state: FieldMappingState) -> FieldMappingState:
    """生成最终输出结果"""
    try:
        final_mapping = state.get("final_mapping", {})
        validation_results = state.get("validation_results", {"passed": True, "issues": []})
        confidence_score = state.get("confidence_score", 0.0)
        
        # 如果final_mapping为空，创建一个默认映射
        if not final_mapping:
            logger.warning("⚠️ final_mapping为空，创建默认映射")
            standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型", 
                "发货方编号", "发货方名称", "收货方编码", "收货方名称", 
                "商品编码", "商品名称", "件数", "体积", "重量"
            ]
            final_mapping = {field: "missing" for field in standard_fields}
        
        # 生成分析依据
        analysis = {
            "依据": f"基于字段名称特征和数据类型分析，共映射了 {len([v for v in final_mapping.values() if v != 'missing'])} 个字段",
            "提醒": validation_results.get("issues", []),
            "准确率": f"{confidence_score}%"
        }
        
        # 组装最终结果
        output = {
            "mapping": final_mapping,
            "analysis": analysis,
            "confidence": int(confidence_score)
        }
        
        state["messages"].append({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)})
        return state
        
    except Exception as e:
        state["errors"].append(f"结果生成失败: {str(e)}")
        return state

# 新增：迭代计数更新节点
def update_iteration_count(state: FieldMappingState) -> FieldMappingState:
    """更新迭代计数"""
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["messages"].append({"role": "system", "content": f"开始第 {state['iteration_count']} 次迭代"})
    return state

# 新增：条件路由函数
def should_retry_mapping(state: FieldMappingState) -> str:
    """判断是否需要重新映射"""
    # 检查验证是否通过
    validation_passed = state.get("validation_results", {}).get("passed", True)
    iteration_count = state.get("iteration_count", 0)
    
    if validation_passed:
        return "generate_output"
    elif iteration_count < 3:  # 最多迭代3次
        return "update_iteration"
    else:
        return "generate_output"

# 新增：创建字段映射图
def create_field_mapping_graph() -> StateGraph:
    """创建字段映射的状态图"""
    try:
        graph_builder = StateGraph(FieldMappingState)
        
        # 添加节点
        graph_builder.add_node("preprocess_fields", preprocess_fields)
        graph_builder.add_node("classify_fields", classify_fields)
        graph_builder.add_node("llm_mapping", llm_map_to_standard_fields)
        graph_builder.add_node("resolve_conflicts", resolve_conflicts_and_missing)
        graph_builder.add_node("validate_mapping", validate_mapping)
        graph_builder.add_node("calculate_confidence", calculate_confidence_score)
        graph_builder.add_node("update_iteration", update_iteration_count)
        graph_builder.add_node("generate_output", generate_output)
        
        # 添加边
        graph_builder.add_edge(START, "preprocess_fields")
        graph_builder.add_edge("preprocess_fields", "classify_fields")
        graph_builder.add_edge("classify_fields", "llm_mapping")
        graph_builder.add_edge("llm_mapping", "resolve_conflicts")
        graph_builder.add_edge("resolve_conflicts", "validate_mapping")
        graph_builder.add_edge("validate_mapping", "calculate_confidence")
        
        # 条件路由：从calculate_confidence决定下一步
        graph_builder.add_conditional_edges(
            "calculate_confidence",
            should_retry_mapping,
            {
                "update_iteration": "update_iteration",
                "generate_output": "generate_output"
            }
        )
        
        # 迭代路径：更新计数后重新映射
        graph_builder.add_edge("update_iteration", "llm_mapping")
        
        # 结束
        graph_builder.add_edge("generate_output", END)
        
        # 设置递归限制
        return graph_builder.compile(checkpointer=None)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create field mapping graph: {str(e)}")

# 创建和配置chatbot的状态图
def create_graph(llm) -> StateGraph:
    try:
        graph_builder = StateGraph(State)
        def chatbot(state: State) -> dict:
            # 处理当前状态并返回 LLM 响应
            return {"messages": [llm.invoke(state["messages"])]}
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        # 这里使用内存存储 也可以持久化到数据库
        memory = MemorySaver()
        # return graph_builder.compile(checkpointer=memory)
        return graph_builder.compile()
    except Exception as e:
        raise RuntimeError(f"Failed to create graph: {str(e)}")

def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logger.info(f"Warning: Failed to save graph visualization: {str(e)}")


# 格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出
def format_response(response):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# # 创建 Nacos 客户端
# nacos_client = nacos.NacosClient(NACOS_SERVER, namespace=NAMESPACE)
# # 停止心跳的标志
# stop_event = threading.Event()
# def heartbeat_loop():
#     while not stop_event.is_set():
#         try:
#             nacos_client.send_heartbeat(SERVICE_NAME, HOST, PORT, group_name=GROUP_NAME)
#             logger.info("💓 心跳发送成功")
#         except Exception as e:
#             logger.info(f"⚠️ 心跳发送失败: {e}")
#         stop_event.wait(HEARTBEAT_INTERVAL)

from src.configs.nacos_helper import NacosHelper
from src.configs.settings import manager, RegisterConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    global graph, field_mapping_graph
    try:
        # 注册服务
        # nacos_client = nacos.NacosClient(NACOS_SERVER, namespace=NAMESPACE)
        # nacos_client.add_naming_instance(SERVICE_NAME, HOST, PORT, group_name=GROUP_NAME)
        # logger.info(f"✅ 已注册到 Nacos: {SERVICE_NAME} @ {HOST}:{PORT}")
        # register_select = manager.get_register_config("nacos")
        # nacos_client = NacosHelper(register_select)
        # heartbeat_thread = threading.Thread(target=nacos_client.heartbeat_loop, daemon=True)
        # nacos_client.register_instance()
        # 启动心跳线程
        # heartbeat_thread.start()
        logger.info("正在初始化模型、定义Graph...")
        graph = create_graph(chat_model)
        # 新增：初始化字段映射图
        field_mapping_graph = create_field_mapping_graph()
        # save_graph_visualization(graph)
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    try:
        yield
    finally:
        print("🛑 停止服务，注销并停止心跳...")
        # nacos_client.stop_event.set()
        # heartbeat_thread.join()
        # nacos_client.deregister()
    # 关闭时执行
    logger.info("正在关闭...")

# lifespan参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)
# 检查nacos服务是否正常 | curl "http://127.0.0.1:8848/nacos/v1/ns/instance/list?serviceName=fastapi-service-greet-system"
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI with Nacos!"}

# 新增：字段映射API端点
@app.post("/field-mapping")
async def field_mapping(request: ChatCompletionRequest):
    """字段映射API：将Excel字段映射到标准字段"""
    if not field_mapping_graph:
        logger.error("字段映射服务未初始化")
        raise HTTPException(status_code=500, detail="字段映射服务未初始化")

    try:
        logger.info(f"收到字段映射请求: {request}")

        # 获取Excel数据
        excel_data = None
        if request.excelData:
            try:
                excel_data = json.loads(request.excelData)
                logger.info(f"使用提供的Excel数据")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Excel数据格式错误")
        elif request.excelFilename:
            try:
                excel_result = excel_processor.read_excel_to_json(request.excelFilename)
                excel_data = excel_result["data"]  # 直接使用行记录列表
                logger.info(f"成功读取Excel文件，共{excel_result['total_rows']}行数据")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Excel文件不存在: {request.excelFilename}")
            except Exception as e:
                logger.error(f"读取Excel文件失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"读取Excel文件失败: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="缺少Excel数据或文件名")

        # 初始化状态
        initial_state = FieldMappingState(
            excel_data=excel_data,
            preprocessed_fields=[],
            classified_fields=[],
            initial_mapping={},
            final_mapping={},
            validation_results={"passed": True, "issues": []},  # 默认验证通过
            confidence_score=0.0,
            iteration_count=0,
            errors=[],
            messages=[]
        )

        # 执行字段映射图
        try:
            # 设置递归限制配置和LangSmith标签
            config = {
                "recursion_limit": 50,
                "tags": ["field-mapping", "excel-processing", "llm-analysis"],
                "metadata": {
                    "excel_filename": request.excelFilename or "json_data",
                    "user_input": request.userInput[:100],  # 截取前100字符
                    "field_count": len(excel_data) if isinstance(excel_data, list) and excel_data else 0
                }
            }
            logger.info("🚀 开始执行字段映射图...")
            result = field_mapping_graph.invoke(initial_state, config=config)
            logger.info("✅ 字段映射图执行完成")
            
            # 提取结果
            final_mapping = result.get("final_mapping", {})
            confidence_score = result.get("confidence_score", 0.0)
            validation_results = result.get("validation_results", {})
            errors = result.get("errors", [])
            
            # 生成响应
            response = {
                "success": True,
                "mapping": final_mapping,
                "confidence_score": confidence_score,
                "validation": validation_results,
                "errors": errors,
                "message": "字段映射完成"
            }
            
            return JSONResponse(content=response)
            
        except Exception as graph_error:
            logger.error(f"字段映射图执行失败: {str(graph_error)}")
            raise HTTPException(status_code=500, detail=f"字段映射执行失败: {str(graph_error)}")

    except Exception as e:
        logger.error(f"处理字段映射请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 封装POST请求接口，与大模型进行问答
@app.post("/greet/stream")
async def chat_completions(request: ChatCompletionRequest):
    # 判断初始化是否完成
    if not graph:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")

        query_prompt = request.userInput

        # query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")
        config = {"configurable": {"thread_id": "456" + "@@" + "456"}}
        # config = {"configurable": {"thread_id": request.userId+"@@"+request.conversationId}}
        logger.info(f"用户当前会话信息: {config}")

        # 根据是否有Excel数据或Excel文件选择不同的提示词
        excel_json_str = None
        
        if request.excelData:
            # 使用请求中的Excel数据
            logger.info("检测到Excel数据，使用test_prompt进行分析")
            excel_json_str = request.excelData
        elif request.excelFilename:
            # 从Excel文件读取数据
            logger.info(f"检测到Excel文件: {request.excelFilename}，使用test_prompt进行分析")
            try:
                excel_data = excel_processor.read_excel_to_json(request.excelFilename)
                excel_json_str = json.dumps(excel_data["data"], ensure_ascii=False)
                logger.info(f"成功读取Excel文件，共{excel_data['total_rows']}行数据")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Excel文件不存在: {request.excelFilename}")
            except Exception as e:
                logger.error(f"读取Excel文件失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"读取Excel文件失败: {str(e)}")
        
        if excel_json_str:
            # 使用test_prompt处理Excel数据
            try:
                system_template = test_prompt.messages[0].prompt.template
                user_template = user_prompt.messages[0].prompt.template
                
                # 格式化系统提示词
                formatted_system_content = system_template.format(excel_data=excel_json_str)
                logger.info(f"系统提示词格式化成功，长度: {len(formatted_system_content)}")
                
                # 格式化用户提示词
                formatted_user_content = user_template.format(query=query_prompt)
                logger.info(f"用户提示词格式化成功，长度: {len(formatted_user_content)}")
                
                prompt = [
                    {"role": "system", "content": formatted_system_content},
                    {"role": "user", "content": formatted_user_content}
                ]
                logger.info("提示词组装成功")
            except Exception as format_error:
                logger.error(f"提示词格式化失败: {str(format_error)}")
                raise HTTPException(status_code=500, detail=f"提示词格式化失败: {str(format_error)}")
        else:
            # 使用greeting_card_prompt进行普通对话
            logger.info("未检测到Excel数据，使用greeting_card_prompt进行普通对话")
            greeting_card_template = greeting_card_prompt.messages[0].prompt.template
            user_template = user_prompt.messages[0].prompt.template

            prompt = [
                {"role": "system", "content": greeting_card_template},
                {"role": "user", "content": user_template.format(query=query_prompt)}
            ]

        # prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS)
        # prompt_template_user = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_USER)
        # prompt = [
        #     {"role": "system", "content": prompt_template_system.template},
        #     {"role": "user", "content": prompt_template_user.template.format(query=query_prompt)}
        # ]

        # 处理流式响应
        if request.stream:
            async def generate_stream():

                try:
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                    async for message_chunk, metadata in graph.astream({"messages": prompt}, config, stream_mode="messages"):
                        try:
                            chunk = message_chunk.content
                            logger.info(f"chunk: {chunk}")

                            # 组装数据
                            # choice = Choice(0, delta={'content': chunk}, finish_reason='null')
                            modelResponseData = ModelResponseData(id=chunk_id,
                                                                  object='chat.completion.chunk',
                                                                  created=current_time,
                                                                  content=chunk,
                                                                  finishReason='null'
                                                                  )
                            modelResponse = ModelResponse(data=modelResponseData)
                            modelResponse = json.loads(modelResponse.to_json())
                            dict_str = {
                                'code': 200,
                                'message': 'success',
                                **modelResponse
                            }
                            json_str = json.dumps(dict_str, ensure_ascii=False)
                            logger.info(f"发送数据json_str:{json_str}")
                            yield f"data: {json_str}\n\n".encode('utf-8')
                        except Exception as chunk_error:
                            # 记录单个数据块处理异常
                            logger.error(f"Error processing stream chunk: {chunk_error}")
                            continue
                        # 原始处理过程：在处理过程中产生每个块
                        # yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {'content': chunk},'finish_reason': None}]})}\n\n"
                    # 产出流结束标记
                    response_data_end = {
                        'code': 200,
                        'message': 'success',
                        'data': {
                            'id': chunk_id,
                            'object': 'chat.completion.chunk',
                            'created': current_time,
                            'content': '',
                            'finishReason': 'stop'
                        }
                    }
                    json_str_end = json.dumps(response_data_end, ensure_ascii=False)
                    logger.info(f"end: {json_str_end}")
                    yield f"data: {json_str_end}\n\n"
                    # 原始处理过程：流结束的最后一块
                    # yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}]})}\n\n"
                except Exception as stream_error:
                    # 记录流生成过程中的异常
                    logger.error(f"Stream generation error: {stream_error}")
                    # 产出错误提示
                    yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"
            # 返回fastapi.responses中StreamingResponse对象
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应处理
        else:
            try:
                events = graph.stream({"messages": prompt}, config)
                for event in events:
                    for value in event.values():
                        result = value["messages"][-1].content
            except Exception as e:
                logger.info(f"Error processing response: {str(e)}")

            formatted_response = str(format_response(result))
            logger.info(f"格式化的搜索结果: {formatted_response}")

            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: \n{response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    logger.info(f"在端口 {app_port} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host=app_host, port=app_port)


