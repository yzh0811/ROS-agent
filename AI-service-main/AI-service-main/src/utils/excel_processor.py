import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """Excel文件处理器，用于将Excel文件转换为JSON格式"""
    
    def __init__(self, excel_dir: str = "excel_files"):
        """
        初始化Excel处理器
        
        Args:
            excel_dir: Excel文件存放目录
        """
        # 使用绝对路径，基于当前文件所在目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # 从 src/utils/excel_processor.py 回到项目根目录
        self.excel_dir = project_root / excel_dir
        self.excel_dir.mkdir(exist_ok=True)
    
    def list_excel_files(self) -> List[str]:
        """列出所有可用的Excel文件"""
        excel_files = []
        if self.excel_dir.exists():
            for file in self.excel_dir.glob("*.xlsx"):
                excel_files.append(file.name)
            for file in self.excel_dir.glob("*.xls"):
                excel_files.append(file.name)
        return excel_files
    
    def read_excel_to_json(self, filename: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        读取Excel文件并转换为JSON格式
        
        Args:
            filename: Excel文件名
            sheet_name: 工作表名称，如果为None则读取所有工作表并合并
            
        Returns:
            包含Excel数据的字典
        """
        try:
            file_path = self.excel_dir / filename
            logger.info(f"尝试读取文件: {file_path}")
            logger.info(f"文件是否存在: {file_path.exists()}")
            logger.info(f"excel_dir路径: {self.excel_dir}")
            logger.info(f"excel_dir是否存在: {self.excel_dir.exists()}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Excel文件不存在: {filename}")
            
            is_xls = file_path.suffix.lower() == '.xls'
            
            # 读取逻辑：
            # - 指定 sheet_name: 只读该表
            # - 未指定: 读取所有工作表，各取前100行后合并，并添加 sheet 列
            MAX_ROWS_PER_SHEET = 100
            
            def _normalize_header(df: pd.DataFrame) -> pd.DataFrame:
                """
                尝试自动识别真实表头：
                - 若当前列名大多为 Unnamed 或空，则在前10行中寻找最可能的表头行
                - 使用该行的值作为列名，并丢弃其之前的行
                - 去除首尾空白、重复列名去重
                """
                try:
                    def _is_unnamed(col: str) -> bool:
                        return isinstance(col, str) and (col.strip() == '' or col.lower().startswith('unnamed'))
                    # 若现有列大多为 Unnamed，则尝试在前10行中寻找表头
                    columns = list(df.columns)
                    unnamed_ratio = sum(1 for c in columns if _is_unnamed(str(c))) / max(1, len(columns))
                    if unnamed_ratio < 0.6:
                        # 大部分列名可用，做基本清洗和去重
                        cleaned = [str(c).strip() if c is not None else '' for c in columns]
                        # 去重
                        seen = {}
                        new_cols = []
                        for c in cleaned:
                            key = c if c != '' else '列'
                            if key not in seen:
                                seen[key] = 0
                                new_cols.append(key)
                            else:
                                seen[key] += 1
                                new_cols.append(f"{key}_{seen[key]}")
                        df.columns = new_cols
                        return df
                    # 在前10行内寻找候选表头行
                    max_rows_scan = min(10, len(df))
                    best_row = None
                    best_score = -1
                    for r in range(max_rows_scan):
                        row = df.iloc[r]
                        # 统计非空字符串的数量作为分数
                        non_empty_text = 0
                        for v in row.values:
                            if isinstance(v, str) and v.strip() != '':
                                non_empty_text += 1
                        score = non_empty_text
                        if score > best_score:
                            best_score = score
                            best_row = r
                    if best_row is not None and best_score > 0:
                        # 采用该行为表头
                        new_cols = []
                        for v in df.iloc[best_row].values:
                            name = str(v).strip() if v is not None else ''
                            if name == '' or _is_unnamed(name):
                                name = '列'
                            new_cols.append(name)
                        # 去重
                        seen = {}
                        dedup_cols = []
                        for c in new_cols:
                            if c not in seen:
                                seen[c] = 0
                                dedup_cols.append(c)
                            else:
                                seen[c] += 1
                                dedup_cols.append(f"{c}_{seen[c]}")
                        df = df.iloc[best_row + 1:].reset_index(drop=True)
                        df.columns = dedup_cols
                        logger.info(f"🔧 表头已重置为第{best_row + 1}行内容: {dedup_cols}")
                    else:
                        # 未找到合适的表头，仅做基本清洗
                        cleaned = [str(c).strip() if c is not None else '' for c in columns]
                        seen = {}
                        new_cols = []
                        for c in cleaned:
                            key = c if c != '' else '列'
                            if key not in seen:
                                seen[key] = 0
                                new_cols.append(key)
                            else:
                                seen[key] += 1
                                new_cols.append(f"{key}_{seen[key]}")
                        df.columns = new_cols
                    return df
                except Exception as _e:
                    logger.warning(f"⚠️ 表头自适应失败，使用原始表头: {str(_e)}")
                    return df
            
            def _process_df(df, sheet_label: str) -> (List[Dict[str, Any]], int, int, bool, List[str]):
                original_rows = len(df)
                truncated = False
                if original_rows > MAX_ROWS_PER_SHEET:
                    df = df.head(MAX_ROWS_PER_SHEET)
                    truncated = True
                records = df.to_dict('records')
                processed_records = []
                for record in records:
                    processed_record = {}
                    for key, value in record.items():
                        if pd.isna(value) or value is None:
                            processed_record[key] = None
                        elif isinstance(value, (int, float)):
                            processed_record[key] = value
                        else:
                            processed_record[key] = str(value)
                    # 添加来源sheet信息
                    processed_record['sheet'] = sheet_label
                    processed_records.append(processed_record)
                return processed_records, original_rows, len(processed_records), truncated, list(df.columns)
            
            sheets_meta = []
            combined_records: List[Dict[str, Any]] = []
            columns_union = set()
            
            if sheet_name:
                try:
                    if is_xls:
                        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='xlrd')
                    else:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                except ImportError as ie:
                    if is_xls:
                        raise ImportError("读取 .xls 文件需要依赖 xlrd，请先安装: pip install xlrd") from ie
                    raise
                # 新增：表头自适应
                df = _normalize_header(df)
                processed_records, original_rows, data_rows, truncated, cols = _process_df(df, sheet_name)
                combined_records.extend(processed_records)
                columns_union.update(cols)
                sheets_meta.append({
                    'name': sheet_name,
                    'original_rows': original_rows,
                    'data_rows': data_rows,
                    'truncated': truncated
                })
            else:
                # 读取所有工作表
                try:
                    excel_file = pd.ExcelFile(file_path, engine='xlrd') if is_xls else pd.ExcelFile(file_path)
                except ImportError as ie:
                    if is_xls:
                        raise ImportError("读取 .xls 文件需要依赖 xlrd，请先安装: pip install xlrd") from ie
                    raise
                total_original_rows = 0
                for sheet in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet, engine='xlrd') if is_xls else pd.read_excel(file_path, sheet_name=sheet)
                    # 新增：表头自适应
                    df = _normalize_header(df)
                    processed_records, original_rows, data_rows, truncated, cols = _process_df(df, sheet)
                    combined_records.extend(processed_records)
                    columns_union.update(cols)
                    total_original_rows += original_rows
                    sheets_meta.append({
                        'name': sheet,
                        'original_rows': original_rows,
                        'data_rows': data_rows,
                        'truncated': truncated
                    })
                # 覆盖单表路径的 original_rows 统计方式
                original_rows = total_original_rows
            
            result = {
                "filename": filename,
                "sheet_name": sheet_name or "ALL",
                "total_rows": original_rows if sheet_name else sum(m['original_rows'] for m in sheets_meta),
                "data_rows": len(combined_records),
                "truncated": any(m['truncated'] for m in sheets_meta),
                "sheets": sheets_meta,
                "columns": sorted(list(columns_union)),
                "data": combined_records
            }
            
            if result["truncated"]:
                logger.info(f"成功读取Excel文件: {filename}, 多表合并，原始共{result['total_rows']}行，合并后取前各{MAX_ROWS_PER_SHEET}行，共{result['data_rows']}行")
            else:
                logger.info(f"成功读取Excel文件: {filename}, 多表合并，原始共{result['total_rows']}行，合并后共{result['data_rows']}行")
            return result
            
        except Exception as e:
            logger.error(f"读取Excel文件失败: {filename}, 错误: {str(e)}")
            raise
    
    def get_excel_info(self, filename: str) -> Dict[str, Any]:
        """
        获取Excel文件的基本信息
        
        Args:
            filename: Excel文件名
            
        Returns:
            Excel文件信息
        """
        try:
            file_path = self.excel_dir / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Excel文件不存在: {filename}")
            
            # 读取Excel文件信息
            excel_file = pd.ExcelFile(file_path)
            
            info = {
                "filename": filename,
                "file_size": file_path.stat().st_size,
                "sheets": excel_file.sheet_names,
                "total_sheets": len(excel_file.sheet_names)
            }
            
            # 获取第一个工作表的基本信息
            if excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])
                info["first_sheet_rows"] = len(df)
                info["first_sheet_columns"] = list(df.columns)
            
            return info
            
        except Exception as e:
            logger.error(f"获取Excel文件信息失败: {filename}, 错误: {str(e)}")
            raise
    
    def save_json_data(self, filename: str, json_data: Dict[str, Any]) -> str:
        """
        将JSON数据保存到文件
        
        Args:
            filename: 保存的文件名
            json_data: 要保存的JSON数据
            
        Returns:
            保存的文件路径
        """
        try:
            json_dir = Path("json_data")
            json_dir.mkdir(exist_ok=True)
            
            json_path = json_dir / f"{filename}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON数据已保存到: {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"保存JSON数据失败: {filename}, 错误: {str(e)}")
            raise

# 创建全局实例
excel_processor = ExcelProcessor() 