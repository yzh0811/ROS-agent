import gradio as gr
import requests
import json
from typing import Dict, Any

# 后端服务地址
API_BASE_URL = "http://localhost:7860"

class FieldMappingUI:
    def __init__(self):
        self.current_progress = 0
        self.is_processing = False
        
    def get_progress(self) -> tuple:
        """获取当前进度并返回UI更新值"""
        try:
            response = requests.get(f"{API_BASE_URL}/field-mapping/progress")
            if response.status_code == 200:
                progress_data = response.json()
                
                if "error" not in progress_data:
                    # 计算进度条值
                    progress_percentage = progress_data.get("progress_percentage", 0)
                    progress_bar_value = progress_percentage / 100
                    
                    # 格式化状态文本
                    completed_steps = progress_data.get("completed_steps", 0)
                    total_steps = progress_data.get("total_steps", 0)
                    current_step_name = progress_data.get("current_step_name", "未知步骤")
                    status_text_value = f"正在处理... ({completed_steps}/{total_steps}) - {current_step_name}"
                    
                    # 格式化进度文本
                    progress_text_value = f"进度: {progress_percentage:.1f}%"
                    
                    # 格式化步骤详情
                    steps_detail_value = self._format_steps_detail(progress_data)
                    
                    return (
                        progress_bar_value,
                        status_text_value,
                        progress_text_value,
                        steps_detail_value
                    )
                else:
                    return (0, f"⚠️ {progress_data['error']}", "0%", "无法获取进度信息")
            else:
                return (0, "⚠️ 无法获取进度信息", "0%", "请求失败")
        except Exception as e:
            return (0, f"❌ 请求失败: {str(e)}", "0%", "连接错误")
    
    def _format_steps_detail(self, progress_data: Dict[str, Any]) -> str:
        """格式化步骤详情"""
        try:
            steps_detail = []
            steps_detail.append("📋 步骤执行详情:")
            
            # 获取所有步骤信息
            steps = progress_data.get("steps", {})
            for step_name, step_info in steps.items():
                if isinstance(step_info, dict):
                    status = step_info.get("status", "unknown")
                    duration = step_info.get("duration", 0)
                    metadata = step_info.get("metadata", {})
                    
                    if status == "completed":
                        status_icon = "✅"
                        status_text = "完成"
                    elif status == "running":
                        status_icon = "🔄"
                        status_text = "执行中"
                    elif status == "failed":
                        status_icon = "❌"
                        status_text = "失败"
                    else:
                        status_icon = "⏳"
                        status_text = "等待"
                    
                    # 格式化元数据
                    meta_text = ""
                    if metadata:
                        meta_items = []
                        for key, value in metadata.items():
                            if isinstance(value, (int, float)):
                                meta_items.append(f"{key}: {value}")
                            elif isinstance(value, str):
                                meta_items.append(f"{key}: {value}")
                        if meta_items:
                            meta_text = f" ({', '.join(meta_items)})"
                    
                    steps_detail.append(f"{status_icon} {step_name}: {status_text} ({duration:.2f}s){meta_text}")
            
            return "\n".join(steps_detail)
        except Exception as e:
            return f"步骤详情格式化失败: {str(e)}"
    

    
    def process_field_mapping(self, excel_file, user_input, max_rows, max_columns):
        """处理字段映射请求"""
        if not excel_file:
            return "请选择Excel文件", None, None
        
        try:
            self.is_processing = True
            
            # 读取Excel文件并转换为JSON
            if hasattr(excel_file, 'name'):
                try:
                    import pandas as pd
                    df = pd.read_excel(excel_file.name)
                    
                    # 数据截断处理
                    original_rows, original_cols = df.shape
                    
                    # 截断行数
                    if len(df) > max_rows:
                        df = df.head(max_rows)
                        print(f"⚠️ 数据行数从 {original_rows} 截断到 {max_rows}")
                    
                    # 截断列数
                    if len(df.columns) > max_columns:
                        df = df.iloc[:, :max_columns]
                        print(f"⚠️ 数据列数从 {original_cols} 截断到 {max_columns}")
                    
                    # 转换为字典格式
                    excel_data = df.to_dict('records')
                    
                    # 添加数据截断信息
                    truncation_info = f"📊 数据处理信息:\n"
                    truncation_info += f"原始数据: {original_rows}行 × {original_cols}列\n"
                    truncation_info += f"处理后: {len(df)}行 × {len(df.columns)}列\n"
                    if original_rows > max_rows or original_cols > max_columns:
                        truncation_info += f"⚠️ 数据已截断以避免token超限\n"
                    
                    # 准备请求数据 - 使用JSON格式
                    json_data = {
                        "excelData": json.dumps(excel_data, ensure_ascii=False),
                        "userInput": user_input or "请帮我映射这些字段",
                        "stream": False
                    }
                    
                    # 发送请求
                    response = requests.post(
                        f"{API_BASE_URL}/field-mapping",
                        json=json_data
                    )
                except Exception as e:
                    return f"❌ 读取Excel文件失败: {str(e)}", None, None
            else:
                return "❌ 文件格式错误", None, None
            
            if response.status_code == 200:
                result = response.json()
                
                # 格式化结果
                if result.get("success"):
                    mapping = result.get("mapping", {})
                    confidence = result.get("confidence_score", 0)
                    
                    # 格式化映射结果
                    mapping_text = "字段映射结果:\n"
                    for standard_field, source_field in mapping.items():
                        if source_field != "missing":
                            mapping_text += f"✅ {standard_field} ← {source_field}\n"
                        else:
                            mapping_text += f"❌ {standard_field} ← 未找到匹配\n"
                    
                    mapping_text += f"\n置信度: {confidence}%"
                    
                    # 进度更新现在由Gradio的every参数自动处理
                    return mapping_text, result, None
                else:
                    return f"❌ 映射失败: {result.get('message', '未知错误')}", None, None
            else:
                return f"❌ 请求失败: {response.status_code}", None, None
                
        except Exception as e:
            self.is_processing = False
            return f"❌ 处理失败: {str(e)}", None, None
    
    def create_ui(self):
        """创建Web界面"""
        with gr.Blocks(title="智能字段映射系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 智能字段映射系统")
            gr.Markdown("上传Excel文件，自动映射到标准运输管理字段")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    excel_file = gr.File(
                        label="📁 选择Excel文件",
                        file_types=[".xlsx", ".xls"],
                        file_count="single"
                    )
                    
                    user_input = gr.Textbox(
                        label="💬 用户输入",
                        placeholder="请描述你的需求，例如：请帮我映射这些字段",
                        value="请帮我映射这些字段"
                    )
                    
                    # 数据截断设置
                    with gr.Row():
                        max_rows = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="最大行数",
                            info="限制处理的行数，避免token超限"
                        )
                        
                        max_columns = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=20,
                            step=5,
                            label="最大列数",
                            info="限制处理的列数"
                        )
                    
                    submit_btn = gr.Button("🚀 开始映射", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    # 进度显示区域
                    gr.Markdown("### 📊 处理进度")
                    
                    self.progress_bar = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0,
                        label="进度条",
                        interactive=False
                    )
                    
                    self.status_text = gr.Textbox(
                        label="状态",
                        value="等待开始...",
                        interactive=False
                    )
                    
                    self.progress_text = gr.Textbox(
                        label="进度详情",
                        value="0%",
                        interactive=False
                    )
                    
                    # 添加步骤详情显示
                    self.steps_detail = gr.Textbox(
                        label="步骤详情",
                        value="等待开始...",
                        lines=5,
                        interactive=False
                    )
            
            # 结果显示区域
            with gr.Row():
                result_text = gr.Textbox(
                    label="📋 映射结果",
                    lines=15,
                    interactive=False,
                    placeholder="映射结果将在这里显示..."
                )
                
                result_json = gr.JSON(
                    label="🔍 详细数据",
                    visible=False
                )
            
            # 错误显示区域
            error_text = gr.Textbox(
                label="⚠️ 错误信息",
                lines=3,
                interactive=False,
                visible=False
            )
            
            # 绑定事件
            submit_btn.click(
                fn=self.process_field_mapping,
                inputs=[excel_file, user_input, max_rows, max_columns],
                outputs=[result_text, result_json, error_text]
            )
            
            # 添加进度更新按钮（手动触发）
            refresh_btn = gr.Button("🔄 刷新进度", size="sm")
            refresh_btn.click(
                fn=self.get_progress,
                outputs=[self.progress_bar, self.status_text, self.progress_text, self.steps_detail]
            )
            
            # 保存引用
            self.progress_bar = self.progress_bar
            self.status_text = self.status_text
            self.progress_text = self.progress_text
            self.steps_detail = self.steps_detail
        
        return demo

def main():
    """主函数"""
    ui = FieldMappingUI()
    demo = ui.create_ui()
    
    # 启动Gradio界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 