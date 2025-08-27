# -*- coding: utf-8 -*-
"""
手动映射规则配置
当LLM无法识别某些字段时，使用这些规则进行补充映射
"""

# 发货方相关映射规则
SHIPPER_MAPPING_RULES = {
    # 发货方编号映射规则
    "发货方编号": [
        "站点编号", "网点编号", "门店编号", "仓库编号", "配送中心编号", "DC编号", "工厂编号",
        "站号", "网号", "店号", "仓号", "厂号", "配送号",
        "站点编码", "网点编码", "门店编码", "仓库编码", "配送中心编码", "DC编码", "工厂编码",
        "站编码", "网编码", "店编码", "仓编码", "厂编码",
        "发货点编号", "始发点编号", "起运点编号", "发运点编号"
    ],
    
    # 发货方名称映射规则
    "发货方名称": [
        "站点名称", "网点名称", "门店名称", "仓库名称", "配送中心名称", "DC名称", "工厂名称",
        "站名", "网名", "店名", "仓名", "厂名", "配送名",
        "发货点名称", "始发点名称", "起运点名称", "发运点名称"
    ]
}

# 收货方相关映射规则
RECEIVER_MAPPING_RULES = {
    # 收货方编号映射规则
    "收货方编号": [
        "送货点编号", "收货点编号", "配送点编号", "目的地编号", "客户编号", "客户号", "终端编号",
        "送货编码", "收货编码", "配送编码", "目的地编码", "客户编码", "终端编码"
    ],
    
    # 收货方名称映射规则
    "收货方名称": [
        "送货点名称", "收货点名称", "配送点名称", "目的地名称", "客户名称", "客户名", "终端名称",
        "送货点", "收货点", "配送点", "目的地", "客户点", "终端"
    ]
}

# 承运商相关映射规则
CARRIER_MAPPING_RULES = {
    "承运商": [
        "运输公司", "物流公司", "快递公司", "配送公司",
        "供应商", "分包商", "指定承运商", "合作承运商"
    ]
}

# 商品相关映射规则
GOODS_MAPPING_RULES = {
    "商品编码": [
        "货号", "品号", "SKU", "sku", "商品代码", "产品代码",
        "Item ID", "Item ID", "Item Code", "Product Code", "Product ID", "Goods Code", "Material Code",
        "ID", "id", "Code", "code", "PID", "pid"
    ],
    "商品名称": [
        "品名", "货名", "商品描述", "产品描述", "货物描述",
        "Item Description", "Item Desc", "Product Name", "Product Description", "Goods Name", "Material Description",
        "Desc", "desc", "Description", "description", "Name", "name"
    ]
}

# 数量字段映射规则
QUANTITY_MAPPING_RULES = {
    "件数": [
        "件数", "数量", "总件数", "件数合计", "件数总计", "箱数", "总箱数", "包数", "总包数",
        "qty", "qty", "count", "pcs", "box", "quantity", "total_qty", "total_count"
    ],
    "体积": [
        "体积", "总体积", "体积合计", "体积总计", "立方", "立方米", "m³", "m3", "容积", "空间", "体积量",
        "volume", "vol", "cbm", "cubic", "total_volume", "volume_total", "cbm_total"
    ],
    "重量": [
        "重量", "总重量", "重量合计", "重量总计", "公斤", "千克", "kg", "KG", "吨", "斤", "磅",
        "weight", "wt", "ton", "lb", "total_weight", "weight_total", "kg_total"
    ]
}

# 其他字段映射规则
OTHER_MAPPING_RULES = {
    "体积": ["立方", "m3", "立方米"],
    "重量": ["公斤", "kg", "吨", "斤"]
}

# 合并所有规则
ALL_MAPPING_RULES = {
    **SHIPPER_MAPPING_RULES,
    **RECEIVER_MAPPING_RULES,
    **CARRIER_MAPPING_RULES,
    **GOODS_MAPPING_RULES,
    **QUANTITY_MAPPING_RULES,
    **OTHER_MAPPING_RULES
}

def get_mapping_rules():
    """获取所有映射规则"""
    return ALL_MAPPING_RULES

def get_shipper_rules():
    """获取发货方相关映射规则"""
    return SHIPPER_MAPPING_RULES

def get_carrier_rules():
    """获取承运商相关映射规则"""
    return CARRIER_MAPPING_RULES

def get_goods_rules():
    """获取商品相关映射规则"""
    return GOODS_MAPPING_RULES 