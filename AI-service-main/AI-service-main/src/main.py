import re
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.configs.model_init import chat_model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from src.schemas.model_response import ModelResponseData, ModelResponse
from src.prompts.prompts import user_prompt, greeting_card_prompt, test_prompt
from src.utils.excel_processor import excel_processor
from datetime import datetime
import sys
import io

# 引入拆分后的图构建模块与API模型、工具
from src.graphs.field_mapping_graph import create_field_mapping_graph as create_field_mapping_graph_ext
from src.graphs.chat_graph import create_graph as create_graph_ext
from src.schemas.api_models import Message, ChatCompletionRequest, ChatCompletionResponseChoice, ChatCompletionResponse
from src.utils.response_formatter import format_response

# 全局对象
graph = None
field_mapping_graph = None
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "AI-service-field-mapping"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
from src.configs.settings import manager, RegisterConfig
service_select = manager.get_service_config("greet-system")
app_host = service_select.app_host
app_port = int(service_select.app_port)


class State(TypedDict):
    messages: Annotated[list, add_messages]

from src.configs.nacos_helper import NacosHelper
from src.configs.settings import manager, RegisterConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, field_mapping_graph
    try:
        logger.info("正在初始化模型、定义Graph...")
        graph = create_graph_ext(chat_model)
        field_mapping_graph = create_field_mapping_graph_ext()
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise
    try:
        yield
    finally:
        print("🛑 停止服务，注销并停止心跳...")
    logger.info("正在关闭...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI with Nacos!"}

@app.post("/field-mapping")
async def field_mapping(request: ChatCompletionRequest):
    if not field_mapping_graph:
        logger.error("字段映射服务未初始化")
        raise HTTPException(status_code=500, detail="字段映射服务未初始化")
    try:
        logger.info(f"收到字段映射请求: {request}")
        excel_data = None
        if request.excelData:
            try:
                excel_data = json.loads(request.excelData)
                logger.info("使用提供的Excel数据")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Excel数据格式错误")
        elif request.excelFilename:
            try:
                excel_result = excel_processor.read_excel_to_json(request.excelFilename)
                excel_data = excel_result["data"]
                logger.info(f"成功读取Excel文件，共{excel_result['total_rows']}行数据")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Excel文件不存在: {request.excelFilename}")
            except Exception as e:
                logger.error(f"读取Excel文件失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"读取Excel文件失败: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="缺少Excel数据或文件名")

        initial_state: Dict[str, Any] = {
            "excel_data": excel_data,
            "preprocessed_fields": [],
            "classified_fields": [],
            "initial_mapping": {},
            "final_mapping": {},
            "validation_results": {"passed": True, "issues": []},
            "confidence_score": 0.0,
            "iteration_count": 0,
            "errors": [],
            "messages": [],
        }
        try:
            config = {
                "recursion_limit": 50,
                "tags": ["field-mapping", "excel-processing", "llm-analysis"],
                "metadata": {
                    "excel_filename": request.excelFilename or "json_data",
                    "user_input": request.userInput[:100],
                    "field_count": len(excel_data) if isinstance(excel_data, list) and excel_data else 0,
                },
            }
            logger.info("🚀 开始执行字段映射图...")
            result = field_mapping_graph.invoke(initial_state, config=config)
            logger.info("✅ 字段映射图执行完成")
            final_mapping = result.get("final_mapping", {})
            confidence_score = result.get("confidence_score", 0.0)
            validation_results = result.get("validation_results", {})
            errors = result.get("errors", [])
            response = {
                "success": True,
                "mapping": final_mapping,
                "confidence_score": confidence_score,
                "validation": validation_results,
                "errors": errors,
                "message": "字段映射完成",
            }
            return JSONResponse(content=response)
        except Exception as graph_error:
            logger.error(f"字段映射图执行失败: {str(graph_error)}")
            raise HTTPException(status_code=500, detail=f"字段映射执行失败: {str(graph_error)}")
    except Exception as e:
        logger.error(f"处理字段映射请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/greet/stream")
async def chat_completions(request: ChatCompletionRequest):
    if not graph:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")
    try:
        logger.info(f"收到聊天完成请求: {request}")
        query_prompt = request.userInput
        config = {"configurable": {"thread_id": "456" + "@@" + "456"}}
        logger.info(f"用户当前会话信息: {config}")

        excel_json_str = None
        if request.excelData:
            logger.info("检测到Excel数据，使用test_prompt进行分析")
            excel_json_str = request.excelData
        elif request.excelFilename:
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
            try:
                system_template = test_prompt.messages[0].prompt.template
                user_template = user_prompt.messages[0].prompt.template
                formatted_system_content = system_template.format(excel_data=excel_json_str)
                logger.info(f"系统提示词格式化成功，长度: {len(formatted_system_content)}")
                formatted_user_content = user_template.format(query=query_prompt)
                logger.info(f"用户提示词格式化成功，长度: {len(formatted_user_content)}")
                prompt = [
                    {"role": "system", "content": formatted_system_content},
                    {"role": "user", "content": formatted_user_content},
                ]
                logger.info("提示词组装成功")
            except Exception as format_error:
                logger.error(f"提示词格式化失败: {str(format_error)}")
                raise HTTPException(status_code=500, detail=f"提示词格式化失败: {str(format_error)}")
        else:
            logger.info("未检测到Excel数据，使用greeting_card_prompt进行普通对话")
            greeting_card_template = greeting_card_prompt.messages[0].prompt.template
            user_template = user_prompt.messages[0].prompt.template
            prompt = [
                {"role": "system", "content": greeting_card_template},
                {"role": "user", "content": user_template.format(query=query_prompt)},
            ]

        if request.stream:
            async def generate_stream():
                try:
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                    async for message_chunk, metadata in graph.astream({"messages": prompt}, config, stream_mode="messages"):
                        try:
                            chunk = message_chunk.content
                            logger.info(f"chunk: {chunk}")
                            modelResponseData = ModelResponseData(
                                id=chunk_id,
                                object='chat.completion.chunk',
                                created=current_time,
                                content=chunk,
                                finishReason='null'
                            )
                            modelResponse = ModelResponse(data=modelResponseData)
                            modelResponse = json.loads(modelResponse.to_json())
                            dict_str = {'code': 200, 'message': 'success', **modelResponse}
                            json_str = json.dumps(dict_str, ensure_ascii=False)
                            logger.info(f"发送数据json_str:{json_str}")
                            yield f"data: {json_str}\n\n".encode('utf-8')
                        except Exception as chunk_error:
                            logger.error(f"Error processing stream chunk: {chunk_error}")
                            continue
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
                except Exception as stream_error:
                    logger.error(f"Stream generation error: {stream_error}")
                    yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
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
                        finish_reason="stop",
                    )
                ]
            )
            logger.info(f"发送响应内容: \n{response}")
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"在端口 {app_port} 上启动服务器")
    uvicorn.run(app, host=app_host, port=app_port)


