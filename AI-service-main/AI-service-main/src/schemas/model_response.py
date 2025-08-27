from dataclasses import dataclass
from dataclasses import asdict
from typing import Dict, List, Literal
from datetime import datetime
import json
from pydantic import BaseModel
from src.logger.logger import logger

@dataclass
class Choice():
    index: int
    # 内容字典，如 {"content": "你好"}
    delta: Dict[str, str]
    # 严格限制为字符串'null'或'stop'
    finish_reason: Literal['null', 'stop']
    def to_dict(self):
        return {'data': self.data}  # 根据实际结构调整


# Python 中的一个装饰器（来自 dataclasses 模块），用于自动为类生成标准方法
@dataclass
class ModelResponseData():
    id: str
    # 固定值
    object: Literal['chat.completion.chunk']
    # 时间字符串
    created: str
    content: str
    finishReason: str
    # choices: List[Choice]
    def to_dict(self):
        return {'data': self.data}  # 根据实际结构调整


@dataclass
class ModelResponse():
    data: ModelResponseData

    def to_dict(self):
        return {'data': self.data}  # 根据实际结构调整


    def to_json(self) -> str:
        return json.dumps({
            "data": {
                "id": self.data.id,
                "object": self.data.object,
                "created": self.data.created,
                "content": self.data.content,
                "finishReason": self.data.finishReason,
            }
        }, ensure_ascii=False)



    # def to_json(self) -> str:
    #     return json.dumps({
    #         "data": {
    #             "id": self.data.id,
    #             "object": self.data.object,
    #             "created": self.data.created,
    #             "choices": [
    #                 {
    #                     "index": c.index,
    #                     "delta": c.delta,
    #                     "finish_reason": c.finish_reason
    #                 } for c in self.data.choices
    #             ]
    #         },
    #     }, ensure_ascii=False)


# 组装数据
# choice = Choice(0, delta={'content': "你好"}, finish_reason='null')
# modelResponseData = ModelResponseData(id='1234',
#                                     object='chat.completion.chunk',
#                                     created='12345',
#                                     choices=[choice],
#                                     )
# modelResponse = ModelResponse(data=modelResponseData)
# logger.info(f"modelResponse.to_json(): {modelResponse.to_json()}")
#
# logger.info(f"modelResponse.to_dict(): {modelResponse.to_dict()}")
# logger.info(f"choice: {choice}")
# logger.info(f"choice-dumps: {json.dumps(choice.__dict__, ensure_ascii=False)}")
# logger.info(f"modelResponse-dumps: {json.dumps(asdict(modelResponse), ensure_ascii=False)}")
# logger.info(f"modelResponse-dumps: {json.dumps(modelResponse.__dict__, ensure_ascii=False)}")
