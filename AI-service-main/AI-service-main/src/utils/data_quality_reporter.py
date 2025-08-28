import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

class DataQualityReporter:
    """数据质量报告生成器"""
    
    def __init__(self):
        self.report_data = {}
    
    def analyze_excel_data(self, excel_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析Excel数据质量"""
        try:
            # 转换数据格式
            if isinstance(excel_data, list) and len(excel_data) > 0:
                if isinstance(excel_data[0], dict):
                    df = pd.DataFrame(excel_data)
                else:
                    df = pd.DataFrame(excel_data)
            else:
                df = pd.DataFrame(excel_data)
            
            # 基础统计
            basic_stats = self._get_basic_stats(df)
            
            # 数据质量指标
            quality_metrics = self._get_quality_metrics(df)
            
            # 字段分析
            field_analysis = self._analyze_fields(df)
            
            # 数据完整性
            completeness = self._analyze_completeness(df)
            
            # 数据一致性
            consistency = self._analyze_consistency(df)
            
            # 生成报告
            report = {
                "timestamp": datetime.now().isoformat(),
                "basic_stats": basic_stats,
                "quality_metrics": quality_metrics,
                "field_analysis": field_analysis,
                "completeness": completeness,
                "consistency": consistency,
                "recommendations": self._generate_recommendations(quality_metrics, completeness, consistency)
            }
            
            self.report_data = report
            return report
            
        except Exception as e:
            return {"error": f"数据分析失败: {str(e)}"}
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取基础统计信息"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "data_types": df.dtypes.to_dict()
        }
    
    def _get_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据质量指标"""
        metrics = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # 空值统计
            null_count = col_data.isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            # 唯一值统计
            unique_count = col_data.nunique()
            unique_percentage = (unique_count / len(df)) * 100
            
            # 数据类型
            data_type = str(col_data.dtype)
            
            # 数值型数据的统计
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    numeric_stats = {
                        "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                        "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                        "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                        "std": float(col_data.std()) if not pd.isna(col_data.std()) else None
                    }
                except:
                    numeric_stats = None
            else:
                numeric_stats = None
            
            metrics[col] = {
                "null_count": int(null_count),
                "null_percentage": round(null_percentage, 2),
                "unique_count": int(unique_count),
                "unique_percentage": round(unique_percentage, 2),
                "data_type": data_type,
                "numeric_stats": numeric_stats
            }
        
        return metrics
    
    def _analyze_fields(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析字段特征"""
        field_analysis = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # 字段名称分析
            field_name = str(col)
            field_name_lower = field_name.lower()
            
            # 字段类型推断
            field_type = self._infer_field_type(field_name_lower, col_data)
            
            # 字段重要性评分
            importance_score = self._calculate_importance_score(field_name_lower, col_data)
            
            field_analysis[col] = {
                "inferred_type": field_type,
                "importance_score": importance_score,
                "keywords": self._extract_keywords(field_name_lower)
            }
        
        return field_analysis
    
    def _infer_field_type(self, field_name: str, col_data: pd.Series) -> str:
        """推断字段类型"""
        # 基于字段名的推断
        if any(keyword in field_name for keyword in ["日期", "时间", "date", "time"]):
            return "date"
        elif any(keyword in field_name for keyword in ["编号", "编码", "code", "id", "号"]):
            return "identifier"
        elif any(keyword in field_name for keyword in ["数量", "件数", "体积", "重量", "count", "volume", "weight"]):
            return "quantity"
        elif any(keyword in field_name for keyword in ["名称", "name", "地址", "address", "公司", "company"]):
            return "name"
        elif any(keyword in field_name for keyword in ["价格", "金额", "price", "amount", "cost"]):
            return "price"
        else:
            return "unknown"
    
    def _calculate_importance_score(self, field_name: str, col_data: pd.Series) -> float:
        """计算字段重要性评分"""
        score = 0.0
        
        # 基于字段名的评分
        if any(keyword in field_name for keyword in ["订单", "运单", "order", "waybill"]):
            score += 0.3
        if any(keyword in field_name for keyword in ["日期", "时间", "date", "time"]):
            score += 0.2
        if any(keyword in field_name for keyword in ["数量", "件数", "count", "qty"]):
            score += 0.2
        if any(keyword in field_name for keyword in ["名称", "name"]):
            score += 0.1
        
        # 基于数据特征的评分
        null_percentage = col_data.isnull().sum() / len(col_data)
        if null_percentage < 0.1:  # 空值率低
            score += 0.1
        if null_percentage < 0.5:  # 空值率中等
            score += 0.05
        
        unique_percentage = col_data.nunique() / len(col_data)
        if 0.1 < unique_percentage < 0.9:  # 唯一性适中
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_keywords(self, field_name: str) -> List[str]:
        """提取字段关键词"""
        keywords = []
        
        # 中文关键词
        chinese_keywords = ["订单", "运单", "日期", "时间", "数量", "件数", "体积", "重量", 
                           "名称", "编号", "编码", "地址", "公司", "价格", "金额"]
        
        # 英文关键词
        english_keywords = ["order", "waybill", "date", "time", "count", "qty", "volume", 
                           "weight", "name", "id", "code", "address", "company", "price", "amount"]
        
        for keyword in chinese_keywords + english_keywords:
            if keyword in field_name:
                keywords.append(keyword)
        
        return keywords
    
    def _analyze_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据完整性"""
        completeness = {}
        
        # 整体完整性
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        overall_completeness = ((total_cells - null_cells) / total_cells) * 100
        
        completeness["overall"] = round(overall_completeness, 2)
        
        # 行完整性
        row_completeness = []
        for idx, row in df.iterrows():
            null_count = row.isnull().sum()
            row_comp = ((len(row) - null_count) / len(row)) * 100
            row_completeness.append(round(row_comp, 2))
        
        completeness["row_stats"] = {
            "min": min(row_completeness),
            "max": max(row_completeness),
            "mean": round(np.mean(row_completeness), 2),
            "std": round(np.std(row_completeness), 2)
        }
        
        # 列完整性
        col_completeness = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            col_comp = ((len(df) - null_count) / len(df)) * 100
            col_completeness[col] = round(col_comp, 2)
        
        completeness["column_stats"] = col_completeness
        
        return completeness
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据一致性"""
        consistency = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # 数据类型一致性
            type_consistency = 1.0
            if len(col_data) > 0:
                expected_type = type(col_data.iloc[0])
                type_matches = sum(1 for val in col_data if type(val) == expected_type)
                type_consistency = type_matches / len(col_data)
            
            # 格式一致性（对于字符串类型）
            format_consistency = 1.0
            if col_data.dtype == 'object':
                # 检查字符串长度的一致性
                str_lengths = [len(str(val)) for val in col_data if pd.notna(val)]
                if str_lengths:
                    avg_length = np.mean(str_lengths)
                    length_variance = np.var(str_lengths)
                    format_consistency = 1 / (1 + length_variance / (avg_length + 1))
            
            consistency[col] = {
                "type_consistency": round(type_consistency, 3),
                "format_consistency": round(format_consistency, 3),
                "overall_consistency": round((type_consistency + format_consistency) / 2, 3)
            }
        
        return consistency
    
    def _generate_recommendations(self, quality_metrics: Dict, completeness: Dict, consistency: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于质量指标的建议
        for field, metrics in quality_metrics.items():
            if metrics["null_percentage"] > 50:
                recommendations.append(f"字段 '{field}' 空值率过高 ({metrics['null_percentage']}%)，建议检查数据源或考虑删除")
            
            if metrics["unique_percentage"] < 5:
                recommendations.append(f"字段 '{field}' 唯一性过低 ({metrics['unique_percentage']}%)，可能包含重复或无效数据")
        
        # 基于完整性的建议
        if completeness["overall"] < 80:
            recommendations.append(f"整体数据完整性较低 ({completeness['overall']}%)，建议检查数据采集流程")
        
        # 基于一致性的建议
        for field, cons in consistency.items():
            if cons["overall_consistency"] < 0.7:
                recommendations.append(f"字段 '{field}' 一致性较低 ({cons['overall_consistency']})，建议统一数据格式")
        
        if not recommendations:
            recommendations.append("数据质量良好，无需特别改进")
        
        return recommendations
    
    def generate_html_report(self) -> str:
        """生成HTML格式的报告"""
        if not self.report_data:
            return "<h1>没有可用的报告数据</h1>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>数据质量报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
                .warning {{ color: #ff6600; }}
                .error {{ color: #cc0000; }}
                .success {{ color: #009900; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 数据质量报告</h1>
                <p>生成时间: {self.report_data.get('timestamp', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>📈 基础统计</h2>
                <div class="metric">总行数: {self.report_data.get('basic_stats', {}).get('total_rows', 'N/A')}</div>
                <div class="metric">总列数: {self.report_data.get('basic_stats', {}).get('total_columns', 'N/A')}</div>
                <div class="metric">内存使用: {self.report_data.get('basic_stats', {}).get('memory_usage', 'N/A')} bytes</div>
            </div>
            
            <div class="section">
                <h2>🔍 数据质量指标</h2>
                <table>
                    <tr>
                        <th>字段名</th>
                        <th>空值率</th>
                        <th>唯一性</th>
                        <th>数据类型</th>
                    </tr>
        """
        
        for field, metrics in self.report_data.get('quality_metrics', {}).items():
            null_class = "warning" if metrics['null_percentage'] > 30 else "success"
            unique_class = "warning" if metrics['unique_percentage'] < 10 else "success"
            
            html += f"""
                    <tr>
                        <td>{field}</td>
                        <td class="{null_class}">{metrics['null_percentage']}%</td>
                        <td class="{unique_class}">{metrics['unique_percentage']}%</td>
                        <td>{metrics['data_type']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>📋 改进建议</h2>
                <ul>
        """
        
        for rec in self.report_data.get('recommendations', []):
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, filename: str = None) -> str:
        """保存报告到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_quality_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            raise Exception(f"保存报告失败: {str(e)}")


# 全局实例
data_quality_reporter = DataQualityReporter() 