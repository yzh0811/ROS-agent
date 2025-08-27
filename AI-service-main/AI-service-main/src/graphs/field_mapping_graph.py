import json
import logging
import re
import unicodedata
from typing import List, Optional, Dict, Any
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class FieldMappingState(TypedDict):
    excel_data: Dict[str, Any]
    preprocessed_fields: List[Dict[str, Any]]
    classified_fields: List[Dict[str, Any]]
    initial_mapping: Dict[str, str]
    final_mapping: Dict[str, str]
    validation_results: Dict[str, Any]
    confidence_score: float
    iteration_count: int
    errors: List[str]
    messages: Annotated[list, add_messages]


def parse_llm_response_robust(result_content: str) -> Dict[str, str]:
    try:
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量"
        ]
        json_patterns = [r'\{.*\}', r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']
        for pattern in json_patterns:
            json_match = re.search(pattern, result_content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    llm_result = json.loads(json_str)
                    if isinstance(llm_result, dict):
                        if "映射结果" in llm_result and isinstance(llm_result["映射结果"], dict):
                            return llm_result["映射结果"]
                        elif "mapping" in llm_result and isinstance(llm_result["mapping"], dict):
                            return llm_result["mapping"]
                        elif any(k in llm_result for k in standard_fields):
                            return {k: llm_result.get(k, "missing") for k in standard_fields}
                        elif "confidence" in llm_result or "准确率" in llm_result:
                            return extract_mapping_from_text(result_content, standard_fields)
                except (json.JSONDecodeError, KeyError):
                    continue
        return extract_mapping_from_text(result_content, standard_fields)
    except Exception as e:
        logger.error(f"LLM响应解析失败: {str(e)}")
        return {}


def apply_similarity_fill(initial_mapping: dict, preprocessed_fields: list, threshold: float = 0.4, only_missing: bool = True) -> dict:
    def _norm(s: str) -> str:
        if s is None:
            return ""
        t = unicodedata.normalize("NFKC", str(s)).lower()
        t = re.sub(r"[_\-\./\\]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _tok(s: str):
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fa5]+", _norm(s))

    def _edit_sim(a: str, b: str) -> float:
        a, b = _norm(a), _norm(b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                )
        dist = dp[la][lb]
        return 1 - dist / max(1, max(la, lb))

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(_tok(a)), set(_tok(b))
        return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

    UNIT_HINTS = {
        "体积": ["cbm", "m3", "m³", "立方", "容积", "cubic", "vol"],
        "重量": ["kg", "公斤", "千克", "吨", "lb", "wt", "weight"],
        "件数": ["pcs", "box", "qty", "count", "箱", "包", "数量", "件"],
    }
    TOTAL_HINTS = ["总", "合计", "总计", "total", "sum"]

    def _bonus(name: str, field: str) -> float:
        n = _norm(name)
        b = 0.0
        if field in UNIT_HINTS and any(h in n for h in UNIT_HINTS[field]):
            b += 0.2
        if field in ["件数", "体积", "重量"] and any(h in n for h in TOTAL_HINTS):
            b += 0.2
        return min(b, 0.4)

    std_fields = [
        "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
        "发货方编号", "发货方名称", "收货方编码", "收货方名称",
        "商品编码", "商品名称", "件数", "体积", "重量",
    ]
    if not preprocessed_fields:
        return initial_mapping
    cols = [f.get("name") for f in preprocessed_fields if f.get("name")]
    used = set(v for v in initial_mapping.values() if v and v != "missing")
    updated = dict(initial_mapping)

    for f in std_fields:
        if only_missing and updated.get(f) and updated[f] != "missing":
            continue
        best_name, best_score = None, 0.0
        for name in cols:
            if name in used:
                continue
            score = 0.6 * _edit_sim(name, f) + 0.4 * _jaccard(name, f) + _bonus(name, f)
            if score > best_score:
                best_score, best_name = score, name
        if best_name and best_score >= threshold:
            updated[f] = best_name
            used.add(best_name)
    return updated


def extract_mapping_from_text(text: str, standard_fields: List[str]) -> Dict[str, str]:
    try:
        mapping = {field: "missing" for field in standard_fields}
        for field in standard_fields:
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
    try:
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量",
        ]
        hybrid_mapping = {field: "missing" for field in standard_fields}
        partial_mapping = extract_mapping_from_text(llm_response, standard_fields)
        rule_mapping = get_rule_based_mapping(state)
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
    try:
        classified_fields = state.get("classified_fields", [])
        standard_fields = [
            "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
            "发货方编号", "发货方名称", "收货方编码", "收货方名称",
            "商品编码", "商品名称", "件数", "体积", "重量",
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


def preprocess_fields(state: FieldMappingState) -> FieldMappingState:
    try:
        excel_data = state["excel_data"]
        preprocessed_fields = []
        if isinstance(excel_data, list) and len(excel_data) > 0:
            if isinstance(excel_data[0], dict):
                columns = list(excel_data[0].keys())
                converted_data = {}
                for col in columns:
                    converted_data[col] = [row.get(col) for row in excel_data]
                excel_data = converted_data
        for column_name, column_data in excel_data.items():
            if not isinstance(column_data, list):
                continue
            total_rows = len(column_data)
            non_null_count = sum(1 for value in column_data if value is not None and value != "")
            null_rate = (total_rows - non_null_count) / total_rows if total_rows > 0 else 1.0
            shipper_keywords = [
                "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",
                "站名", "网名", "店名", "仓名", "厂名",
                "站号", "网号", "店号", "仓号", "厂号",
                "站编码", "网编码", "店编码", "仓编码", "厂编码",
                "发货", "发送", "始发", "起运", "发运",
            ]
            is_shipper_field = any(keyword in column_name for keyword in shipper_keywords)
            if null_rate > 0.9 and not is_shipper_field:
                continue
            elif is_shipper_field and null_rate > 0.9:
                logger.info(f"🔍 [预处理] 发货方字段被保护: {column_name} (空值率: {null_rate:.2%})")
            unique_values = set(str(v) for v in column_data if v is not None and v != "")
            uniqueness = len(unique_values) / total_rows if total_rows > 0 else 0
            data_type = "text"
            if all(isinstance(v, (int, float)) for v in column_data if v is not None):
                data_type = "numeric"
            elif all(isinstance(v, str) and len(v) == 10 for v in column_data if v is not None):
                data_type = "date"
            field_info = {
                "name": column_name,
                "data": column_data,
                "null_rate": null_rate,
                "uniqueness": uniqueness,
                "data_type": data_type,
                "total_rows": total_rows,
                "non_null_count": non_null_count,
            }
            preprocessed_fields.append(field_info)
        state["preprocessed_fields"] = preprocessed_fields
        state["messages"].append({"role": "system", "content": f"字段预处理完成，共处理 {len(preprocessed_fields)} 个有效字段"})
        return state
    except Exception as e:
        state["errors"].append(f"字段预处理失败: {str(e)}")
        return state


def classify_fields(state: FieldMappingState) -> FieldMappingState:
    try:
        preprocessed_fields = state["preprocessed_fields"]
        classified_fields = []
        for field in preprocessed_fields:
            field_name = field["name"].lower()
            data_type = field["data_type"]
            uniqueness = field["uniqueness"]
            category = "unknown"
            if any(keyword in field_name for keyword in ["时间", "日期", "date", "time"]) or data_type == "date":
                category = "date"
            elif any(keyword in field_name for keyword in ["编号", "编码", "code", "id", "号"]) and uniqueness > 0.8:
                category = "identifier"
            elif any(keyword in field_name for keyword in ["数量", "件数", "体积", "重量", "count", "volume", "weight"]) and data_type == "numeric":
                category = "quantity"
            elif any(keyword in field_name for keyword in ["名称", "name", "地址", "address", "公司", "company"]):
                category = "name"
            field["category"] = category
            classified_fields.append(field)
        state["classified_fields"] = classified_fields
        state["messages"].append({"role": "system", "content": f"字段分类完成，共分类 {len(classified_fields)} 个字段"})
        return state
    except Exception as e:
        state["errors"].append(f"字段分类失败: {str(e)}")
        return state


def llm_map_to_standard_fields(state: FieldMappingState) -> FieldMappingState:
    try:
        from src.prompts.prompts import test_prompt
        from src.configs.model_init import chat_model
        excel_data = state["excel_data"]
        excel_json_str = json.dumps(excel_data, ensure_ascii=False)
        try:
            system_template = test_prompt.messages[0].prompt.template
            formatted_system_content = system_template.format(excel_data=excel_json_str)
            prompt = [{"role": "system", "content": formatted_system_content}]
            response = chat_model.invoke(prompt)
            result_content = response.content
            try:
                initial_mapping = parse_llm_response_robust(result_content)
                if initial_mapping:
                    state["initial_mapping"] = initial_mapping
                    state["confidence_score"] = 85.0
                    state["messages"].append({
                        "role": "assistant",
                        "content": f"LLM字段映射完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段，置信度: 85%",
                    })
                    try:
                        pre_fields = state.get("preprocessed_fields", [])
                        sim_updated = apply_similarity_fill(state["initial_mapping"], pre_fields, threshold=0.4, only_missing=True)
                        if sim_updated != state["initial_mapping"]:
                            state["initial_mapping"] = sim_updated
                    except Exception:
                        pass
                else:
                    initial_mapping = try_hybrid_mapping(state, result_content)
                    state["initial_mapping"] = initial_mapping
                    state["messages"].append({
                        "role": "system",
                        "content": f"混合策略完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段",
                    })
            except Exception as parse_error:
                state["errors"].append(f"LLM响应处理失败: {str(parse_error)}")
                return rule_based_mapping_fallback(state)
        except Exception as llm_error:
            state["errors"].append(f"LLM调用失败: {str(llm_error)}")
            return rule_based_mapping_fallback(state)
        return state
    except Exception as e:
        state["errors"].append(f"LLM字段匹配失败: {str(e)}")
        return rule_based_mapping_fallback(state)


def rule_based_mapping_fallback(state: FieldMappingState) -> FieldMappingState:
    try:
        logger.info("🔍 [规则降级] 开始规则匹配降级")
        initial_mapping = get_rule_based_mapping(state)
        try:
            from src.configs.manual_mapping_rules import get_mapping_rules
            mapping_rules = get_mapping_rules()
            preprocessed_fields = state.get("preprocessed_fields", [])
            available_fields = [field["name"] for field in preprocessed_fields]
            logger.info(f"🔍 [规则回退] 可用字段: {available_fields}")
            for standard_field, keywords in mapping_rules.items():
                if initial_mapping.get(standard_field) == "missing":
                    for field_name in available_fields:
                        if any(keyword in field_name for keyword in keywords):
                            initial_mapping[standard_field] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充 {standard_field}: {field_name}")
                            break
        except ImportError:
            logger.warning("⚠️ [规则回退] 无法导入手动映射规则，使用内置规则")
            preprocessed_fields = state.get("preprocessed_fields", [])
            if initial_mapping.get("发货方编号") == "missing" or initial_mapping.get("发货方名称") == "missing":
                logger.info("🔍 [规则回退] 尝试智能补充发货方映射...")
                shipper_id_keywords = ["站点编号", "网点编号", "门店编号", "仓库编号", "配送中心编号", "DC编号", "工厂编号", "站号", "网号", "店号", "仓号", "厂号"]
                shipper_name_keywords = ["站点名称", "网点名称", "门店名称", "仓库名称", "配送中心名称", "DC名称", "工厂名称", "站名", "网名", "店名", "仓名", "厂名"]
                if initial_mapping.get("发货方编号") == "missing":
                    for field in preprocessed_fields:
                        field_name = field["name"]
                        if any(keyword in field_name for keyword in shipper_id_keywords):
                            initial_mapping["发货方编号"] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充发货方编号: {field_name}")
                            break
                if initial_mapping.get("发货方名称") == "missing":
                    for field in preprocessed_fields:
                        field_name = field["name"]
                        if any(keyword in field_name for keyword in shipper_name_keywords):
                            initial_mapping["发货方名称"] = field_name
                            logger.info(f"🔍 [规则回退] 智能补充发货方名称: {field_name}")
                            break
        try:
            pre_fields = state.get("preprocessed_fields", [])
            sim_updated = apply_similarity_fill(initial_mapping, pre_fields, threshold=0.4, only_missing=True)
            if sim_updated != initial_mapping:
                initial_mapping = sim_updated
        except Exception:
            pass
        state["initial_mapping"] = initial_mapping
        state["messages"].append({"role": "system", "content": f"规则匹配降级完成，映射了 {len([v for v in initial_mapping.values() if v != 'missing'])} 个字段"})
        return state
    except Exception as e:
        state["errors"].append(f"规则匹配降级失败: {str(e)}")
        return state


def calculate_mapping_score(standard_field: str, field: Dict[str, Any]) -> float:
    score = 0.0
    field_name = field["name"].lower()
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
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "体积":
        if any(keyword in field_name for keyword in ["体积", "volume", "立方", "m3", "总体积", "体积合计", "体积总计", "容积", "空间", "体积量", "vol", "cbm", "cubic"]):
            score += 0.8
        if field["data_type"] == "numeric":
            score += 0.2
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "重量":
        if any(keyword in field_name for keyword in ["重量", "weight", "公斤", "kg", "重", "总重量", "重量合计", "重量总计", "千克", "吨", "斤", "磅", "wt", "ton", "lb"]):
            score += 0.8
        if field["data_type"] == "numeric":
            score += 0.2
        if any(keyword in field_name for keyword in ["总", "合计", "总计", "total", "sum"]):
            score += 0.3
    elif standard_field == "承运商":
        if any(keyword in field_name for keyword in [
            "承运商", "物流", "运输", "carrier", "logistics",
            "运输公司", "物流公司", "快递公司", "配送公司",
            "供应商", "分包商", "指定承运商", "合作承运商",
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
        if any(keyword in field_name for keyword in [
            "发货", "发送", "sender", "shipper", "发",
            "站点", "网点", "门店", "仓库", "配送中心", "DC", "工厂",
            "站名", "网名", "店名", "仓名", "厂名",
            "发货点", "始发点", "起运点", "发运点",
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
        if any(keyword in field_name for keyword in ["发货", "发送", "sender", "shipper", "发"]) and any(keyword in field_name for keyword in ["编号", "编码", "code", "id"]):
            score += 0.8
        elif any(keyword in field_name for keyword in [
            "站点编号", "网点编号", "门店编号", "仓库编号", "配送中心编号", "DC编号", "工厂编号",
            "站号", "网号", "店号", "仓号", "厂号", "配送号",
        ]):
            score += 0.9
        elif any(keyword in field_name for keyword in [
            "站点编码", "网点编码", "门店编码", "仓库编码", "配送中心编码", "DC编码", "工厂编码",
            "站编码", "网编码", "店编码", "仓编码", "厂编码",
        ]):
            score += 0.9
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "收货方编码":
        if any(keyword in field_name for keyword in ["收货", "接收", "receiver"]) and any(keyword in field_name for keyword in ["编号", "编码", "code", "id"]):
            score += 0.8
        elif any(keyword in field_name for keyword in [
            "送货点", "送货点编号", "配送点", "配送点编号", "目的地", "目的地编号", "客户", "客户编号", "客户号", "终端", "终端编号",
        ]):
            score += 0.9
        if any(keyword in field_name.lower() for keyword in ["送货", "收货", "配送", "目的地", "客户", "终端"]):
            score += 0.3
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "收货方名称":
        if any(keyword in field_name for keyword in ["收货", "接收", "receiver"]) and any(keyword in field_name for keyword in ["名称", "name", "收货方名称"]):
            score += 0.8
        elif any(keyword in field_name for keyword in [
            "送货点", "送货点名称", "配送点", "配送点名称", "目的地", "目的地名称", "客户", "客户名称", "客户名", "终端", "终端名称",
        ]):
            score += 0.9
        if any(keyword in field_name.lower() for keyword in ["送货", "收货", "配送", "目的地", "客户", "终端"]):
            score += 0.3
        if field["category"] == "name":
            score += 0.2
    elif standard_field == "商品编码":
        if any(keyword in field_name for keyword in ["商品", "产品", "product", "goods", "货物", "货品"]) and any(keyword in field_name for keyword in ["编号", "编码", "code", "id", "货号", "品号"]):
            score += 0.8
        elif any(keyword in field_name for keyword in [
            "货号", "品号", "SKU", "sku", "商品代码", "产品代码", "item id", "item id", "item code", "product code", "product id", "goods code", "material code",
        ]):
            score += 0.9
        if any(keyword in field_name.lower() for keyword in ["item", "product", "goods", "material", "sku"]):
            score += 0.3
        if field["category"] == "identifier":
            score += 0.2
    elif standard_field == "商品名称":
        if any(keyword in field_name for keyword in ["商品", "产品", "product", "goods", "货物", "货品"]) and any(keyword in field_name for keyword in ["名称", "name", "品名", "货名"]):
            score += 0.8
        elif any(keyword in field_name for keyword in [
            "品名", "货名", "商品描述", "产品描述", "货物描述", "item description", "item desc", "product name", "product description", "goods name", "material description",
        ]):
            score += 0.9
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


def resolve_conflicts_and_missing(state: FieldMappingState) -> FieldMappingState:
    try:
        initial_mapping = state.get("initial_mapping", {})
        logger.info(f"🔍 [冲突处理] 开始处理冲突与缺失")
        logger.info(f"🔍 [冲突处理] 冲突处理前的映射: {json.dumps(initial_mapping, ensure_ascii=False)}")
        if not initial_mapping:
            logger.warning("⚠️ initial_mapping为空，创建默认映射")
            standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
                "发货方编号", "发货方名称", "收货方编码", "收货方名称",
                "商品编码", "商品名称", "件数", "体积", "重量",
            ]
            initial_mapping = {field: "missing" for field in standard_fields}
        final_mapping = initial_mapping.copy()
        value_counts = {}
        for standard_field, personalized_field in initial_mapping.items():
            if personalized_field != "missing":
                if personalized_field in value_counts:
                    value_counts[personalized_field].append(standard_field)
                else:
                    value_counts[personalized_field] = [standard_field]
        if value_counts:
            logger.info(f"🔍 [冲突处理] 检测到的字段映射关系:")
            for personalized_field, standard_fields in value_counts.items():
                if len(standard_fields) > 1:
                    logger.info(f"🔍 [冲突处理] 冲突: {personalized_field} -> {standard_fields}")
                else:
                    logger.info(f"🔍 [冲突处理] 正常: {personalized_field} -> {standard_fields}")
        else:
            logger.info(f"🔍 [冲突处理] 没有检测到任何字段映射关系")
        conflict_count = 0
        for personalized_field, standard_fields in value_counts.items():
            if len(standard_fields) > 1:
                conflict_count += 1
                logger.info(f"🔍 [冲突处理] 处理第{conflict_count}个冲突: {personalized_field} -> {standard_fields}")
                priority_order = ["件数", "体积", "重量", "商品编码", "商品名称", "收货方编码", "收货方名称", "发货方编号", "发货方名称", "运输日期", "订单号", "运单号"]
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
        logger.info(f"🔍 [冲突处理] 冲突处理后的映射: {json.dumps(final_mapping, ensure_ascii=False)}")
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


def validate_mapping(state: FieldMappingState) -> FieldMappingState:
    try:
        final_mapping = state["final_mapping"]
        validation_results = {"passed": True, "issues": []}
        important_fields = ["订单号", "运单号", "运输日期"]
        missing_important = [field for field in important_fields if final_mapping.get(field) == "missing"]
        if missing_important:
            validation_results["issues"].append(f"建议补充重要字段: {', '.join(missing_important)}")
        if final_mapping.get("路顺") != "missing" and final_mapping.get("运单号") == "missing":
            validation_results["issues"].append("路顺字段存在但缺少运单号")
        quantity_fields = ["件数", "体积", "重量"]
        if all(final_mapping.get(field) == "missing" for field in quantity_fields):
            validation_results["issues"].append("缺少所有数量相关字段")
        validation_results["passed"] = True
        state["validation_results"] = validation_results
        state["messages"].append({"role": "system", "content": f"映射验证完成，通过: {validation_results['passed']}"})
        return state
    except Exception as e:
        state["errors"].append(f"映射验证失败: {str(e)}")
        return state


def calculate_confidence_score(state: FieldMappingState) -> FieldMappingState:
    try:
        final_mapping = state["final_mapping"]
        total_fields = len(final_mapping)
        mapped_fields = len([v for v in final_mapping.values() if v != "missing"])
        coverage_rate = mapped_fields / total_fields if total_fields > 0 else 0
        important_fields = ["订单号", "运单号", "运输日期", "件数", "体积", "重量"]
        other_fields = ["路顺", "承运商", "车型", "发货方名称", "收货方名称", "商品名称"]
        important_score = sum(1 for field in important_fields if final_mapping.get(field) != "missing") / len(important_fields)
        other_score = sum(1 for field in other_fields if final_mapping.get(field) != "missing") / len(other_fields)
        validation_score = 1.0
        confidence_score = (0.6 * important_score + 0.3 * other_score + 0.1 * validation_score)
        confidence_score = round(confidence_score * 100, 1)
        state["confidence_score"] = confidence_score
        state["messages"].append({"role": "system", "content": f"准确率评分完成: {confidence_score}%"})
        return state
    except Exception as e:
        state["errors"].append(f"准确率评分失败: {str(e)}")
        return state


def generate_output(state: FieldMappingState) -> FieldMappingState:
    try:
        final_mapping = state.get("final_mapping", {})
        validation_results = state.get("validation_results", {"passed": True, "issues": []})
        confidence_score = state.get("confidence_score", 0.0)
        if not final_mapping:
            logger.warning("⚠️ final_mapping为空，创建默认映射")
            standard_fields = [
                "运输日期", "订单号", "路顺", "承运商", "运单号", "车型",
                "发货方编号", "发货方名称", "收货方编码", "收货方名称",
                "商品编码", "商品名称", "件数", "体积", "重量",
            ]
            final_mapping = {field: "missing" for field in standard_fields}
        analysis = {
            "依据": f"基于字段名称特征和数据类型分析，共映射了 {len([v for v in final_mapping.values() if v != 'missing'])} 个字段",
            "提醒": validation_results.get("issues", []),
            "准确率": f"{confidence_score}%",
        }
        output = {"mapping": final_mapping, "analysis": analysis, "confidence": int(confidence_score)}
        state["messages"].append({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)})
        return state
    except Exception as e:
        state["errors"].append(f"结果生成失败: {str(e)}")
        return state


def update_iteration_count(state: FieldMappingState) -> FieldMappingState:
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["messages"].append({"role": "system", "content": f"开始第 {state['iteration_count']} 次迭代"})
    return state


def should_retry_mapping(state: FieldMappingState) -> str:
    validation_passed = state.get("validation_results", {}).get("passed", True)
    iteration_count = state.get("iteration_count", 0)
    if validation_passed:
        return "generate_output"
    elif iteration_count < 3:
        return "update_iteration"
    else:
        return "generate_output"


def create_field_mapping_graph() -> StateGraph:
    try:
        graph_builder = StateGraph(FieldMappingState)
        graph_builder.add_node("preprocess_fields", preprocess_fields)
        graph_builder.add_node("classify_fields", classify_fields)
        graph_builder.add_node("llm_mapping", llm_map_to_standard_fields)
        graph_builder.add_node("resolve_conflicts", resolve_conflicts_and_missing)
        graph_builder.add_node("validate_mapping", validate_mapping)
        graph_builder.add_node("calculate_confidence", calculate_confidence_score)
        graph_builder.add_node("update_iteration", update_iteration_count)
        graph_builder.add_node("generate_output", generate_output)
        graph_builder.add_edge(START, "preprocess_fields")
        graph_builder.add_edge("preprocess_fields", "classify_fields")
        graph_builder.add_edge("classify_fields", "llm_mapping")
        graph_builder.add_edge("llm_mapping", "resolve_conflicts")
        graph_builder.add_edge("resolve_conflicts", "validate_mapping")
        graph_builder.add_edge("validate_mapping", "calculate_confidence")
        graph_builder.add_conditional_edges(
            "calculate_confidence",
            should_retry_mapping,
            {"update_iteration": "update_iteration", "generate_output": "generate_output"},
        )
        graph_builder.add_edge("update_iteration", "llm_mapping")
        graph_builder.add_edge("generate_output", END)
        return graph_builder.compile(checkpointer=None)
    except Exception as e:
        raise RuntimeError(f"Failed to create field mapping graph: {str(e)}") 