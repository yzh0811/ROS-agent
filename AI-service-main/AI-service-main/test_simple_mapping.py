#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试回退后的简单字段映射图
"""

import requests
import json
import time

def test_field_mapping():
    """测试字段映射接口"""
    print("🧪 测试回退后的字段映射接口")
    print("=" * 50)
    
    # 测试数据
    test_data = {
        "userInput": "请分析这份物流运输数据，识别各个字段的含义并进行标准化映射",
        "stream": False,
        "userId": "test_user",
        "conversationId": "test_conv",
        "excelFilename": "华润万家——5月28日回单.xlsx"
    }
    
    try:
        # 发送请求
        print("📤 发送字段映射请求...")
        response = requests.post(
            "http://localhost:7860/field-mapping",
            json=test_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 请求成功!")
            print(f"📊 响应状态: {result.get('status', 'unknown')}")
            
            # 检查映射结果
            mapping = result.get('mapping', {})
            if mapping:
                print(f"🎯 映射字段数: {len(mapping)}")
                print("📋 映射详情:")
                for standard_field, personalized_field in mapping.items():
                    print(f"  {standard_field} -> {personalized_field}")
                
                # 检查是否有空映射
                missing_count = sum(1 for v in mapping.values() if v == "missing")
                print(f"⚠️ 缺失字段数: {missing_count}")
                
                if missing_count == len(mapping):
                    print("❌ 所有字段都映射为missing，存在问题！")
                else:
                    print("✅ 映射结果正常")
            else:
                print("❌ 映射结果为空")
                
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请先启动服务: python src/main.py")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    print("🚀 开始测试回退后的字段映射图")
    print("请确保服务已启动: python src/main.py")
    print()
    
    # 等待服务启动
    print("⏳ 等待3秒让服务完全启动...")
    time.sleep(3)
    
    # 测试回退后的接口
    test_field_mapping()
    
    print("\n🏁 测试完成!") 