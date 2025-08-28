#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动Web界面的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("🚀 启动智能字段映射Web界面...")
    print("=" * 50)
    
    try:
        # 检查依赖
        try:
            import gradio as gr
            print("✅ Gradio已安装")
        except ImportError:
            print("❌ Gradio未安装，正在安装...")
            os.system("pip install gradio")
            import gradio as gr
            print("✅ Gradio安装完成")
        
        # 导入Web界面
        from src.web.field_mapping_ui import FieldMappingUI
        
        print("✅ Web界面模块导入成功")
        
        # 创建界面
        ui = FieldMappingUI()
        demo = ui.create_ui()
        
        print("✅ Web界面创建成功")
        print("\n🌐 启动信息:")
        print(f"   本地访问: http://localhost:7863")
        print(f"   网络访问: http://0.0.0.0:7863")
        print(f"   按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 启动界面
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7863,        # 使用端口7863避免冲突
            share=False,             # 不分享到公网
            show_error=True,         # 显示错误信息
            quiet=False              # 显示启动信息
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 可能的解决方案:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 检查端口7863是否被占用")
        print("3. 尝试使用其他端口")
        print("4. 检查防火墙设置")


if __name__ == "__main__":
    main()
