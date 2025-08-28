import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProgressStep:
    """进度步骤"""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.current_step = 0
        self.steps: Dict[str, ProgressStep] = {}
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.callbacks: Dict[str, Callable] = {}
        
        # 预定义步骤
        self.define_steps()
    
    def define_steps(self):
        """定义进度步骤"""
        self.steps = {
            "preprocess_fields": ProgressStep("字段预处理"),
            "classify_fields": ProgressStep("字段分类"),
            "llm_mapping": ProgressStep("LLM智能映射"),
            "resolve_conflicts": ProgressStep("冲突解决"),
            "validate_mapping": ProgressStep("映射验证"),
            "calculate_confidence": ProgressStep("置信度计算"),
            "generate_output": ProgressStep("结果生成")
        }
        self.total_steps = len(self.steps)
    
    def start_step(self, step_name: str) -> ProgressStep:
        """开始一个步骤"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = "running"
            step.start_time = datetime.now()
            self.current_step += 1
            
            logger.info(f"🚀 开始步骤: {step_name} ({self.current_step}/{self.total_steps})")
            self._notify_callbacks("step_started", step_name, step)
            return step
        else:
            logger.warning(f"未知步骤: {step_name}")
            return ProgressStep(step_name, "unknown")
    
    def complete_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> ProgressStep:
        """完成一个步骤"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = "completed"
            step.end_time = datetime.now()
            if step.start_time:
                step.duration = (step.end_time - step.start_time).total_seconds()
            
            if metadata:
                step.metadata.update(metadata)
            
            logger.info(f"✅ 完成步骤: {step_name} (耗时: {step.duration:.2f}s)")
            self._notify_callbacks("step_completed", step_name, step)
            return step
        else:
            logger.warning(f"未知步骤: {step_name}")
            return ProgressStep(step_name, "unknown")
    
    def fail_step(self, step_name: str, error: str, metadata: Optional[Dict[str, Any]] = None) -> ProgressStep:
        """标记步骤失败"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = "failed"
            step.end_time = datetime.now()
            step.error = error
            
            if step.start_time:
                step.duration = (step.end_time - step.start_time).total_seconds()
            
            if metadata:
                step.metadata.update(metadata)
            
            logger.error(f"❌ 步骤失败: {step_name} - {error}")
            self._notify_callbacks("step_failed", step_name, step)
            return step
        else:
            logger.warning(f"未知步骤: {step_name}")
            return ProgressStep(step_name, "unknown")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        completed_steps = sum(1 for step in self.steps.values() if step.status == "completed")
        failed_steps = sum(1 for step in self.steps.values() if step.status == "failed")
        running_steps = sum(1 for step in self.steps.values() if step.status == "running")
        
        total_duration = None
        if self.end_time:
            total_duration = (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            total_duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_steps": self.total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "running_steps": running_steps,
            "pending_steps": self.total_steps - completed_steps - failed_steps - running_steps,
            "progress_percentage": (completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0,
            "total_duration": total_duration,
            "current_step": self.current_step,
            "steps": {name: {
                "status": step.status,
                "duration": step.duration,
                "error": step.error,
                "metadata": step.metadata
            } for name, step in self.steps.items()}
        }
    
    def get_step_status(self, step_name: str) -> Optional[ProgressStep]:
        """获取特定步骤状态"""
        return self.steps.get(step_name)
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _notify_callbacks(self, event: str, step_name: str, step: ProgressStep):
        """通知回调函数"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(event, step_name, step)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {str(e)}")
    
    def finish(self):
        """完成整个流程"""
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        completed_steps = sum(1 for step in self.steps.values() if step.status == "completed")
        success_rate = (completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        
        logger.info(f"🎉 流程完成! 成功率: {success_rate:.1f}%, 总耗时: {total_duration:.2f}s")
        
        return {
            "success": completed_steps == self.total_steps,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "completed_steps": completed_steps,
            "total_steps": self.total_steps
        }
    
    def reset(self):
        """重置进度跟踪器"""
        self.current_step = 0
        self.start_time = datetime.now()
        self.end_time = None
        
        for step in self.steps.values():
            step.status = "pending"
            step.start_time = None
            step.end_time = None
            step.duration = None
            step.error = None
            step.metadata.clear()
        
        logger.info("🔄 进度跟踪器已重置")


# 全局进度跟踪器实例
progress_tracker = ProgressTracker() 