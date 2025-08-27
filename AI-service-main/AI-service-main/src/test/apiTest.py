import requests
import json
import logging

import sys
import io

from sympy.physics.paulialgebra import delta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')





# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


url = "http://localhost:7860/greet/stream"
headers = {"Content-Type": "application/json; charset=utf-8"}


# 默认非流式输出 True or False
stream_flag = True

input_text = "朋友聚会"
# input_text = "200元以下，流量大的套餐有啥"
# input_text = "就上面提到的这个套餐，是多少钱"
# input_text = "你说那个10G的套餐，叫啥名字"
# input_text = "你说那个100000000G的套餐，叫啥名字"

# 封装请求的参数
data = {
    "userInput": input_text
}





# data = {
#     # "messages": [{"role": "user", "content": input_text}],
#     "userInput": input_text,
#     # "stream": stream_flag,
#     # "userId":"456",
#     # "conversationId":"456"
# }


# 接收流式输出处理
if stream_flag:
    full_response = ""
    try:
        with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
            for line in response.iter_lines():
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    # 检查是否为空或不合法的字符串
                    if not json_str:
                        logger.info(f"收到空字符串，跳过...")
                        continue
                    # 确保字符串是有效的JSON格式
                    if json_str.startswith('{') and json_str.endswith('}'):
                        try:
                            data = json.loads(json_str)
                            if data['data'].get('finishReason') == "null":
                                delta_content = data['data'].get('content', '')
                                logger.info(f"流式输出，响应部分是: {delta_content}")
                                full_response += delta_content
                            if data['data'].get('finishReason') == "stop":
                                logger.info(f"接收JSON数据结束")
                                logger.info(f"完整响应是: {full_response}")
                            # if 'delta' in data['data']['choices'][0]:
                            #     delta_content = data['data']['choices'][0]['delta'].get('content', '')
                            #     full_response += delta_content
                            #     logger.info(f"流式输出，响应部分是: {delta_content}")
                            # if data['data']['choices'][0].get('finish_reason') == "stop":
                            #     logger.info(f"接收JSON数据结束")
                            #     logger.info(f"完整响应是: {full_response}")
                        except json.JSONDecodeError as e:
                            logger.info(f"JSON解析错误: {e}")
                    else:
                        logger.info(f"无效JSON格式: {json_str}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

# 接收非流式输出处理
else:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        # 强制设置返回内容编码
        response.encoding = 'utf-8'
        # 输出原始 bytes 内容（不解码）
        logger.info(f"原始响应（二进制）: {response.content}")
        # 输出解码后的纯文本（看看是什么格式）
        logger.info(f"原始响应（文本）: {response.text}")
        # 再尝试 JSON 解析
        response_json = response.json()
        logger.info(f"接收到返回的响应原始内容: {response_json}\n")

        content = response.json()['choices'][0]['message']['content']
        logger.info(f"非流式输出，响应内容是: {content}\n")
    except Exception as e:
        logger.error(f"解析失败: {e}")
        logger.error(f"response.content: {response.content}")
        logger.error(f"response.text: {response.text}")

if __name__ == '__main__':
    # try:
    #     # data_v1 =json.dumps(data)
    #     # response = requests.post(url, headers=headers, data=data_v1)
    #
    #     # logger.info(f"response:{response}")
    # except Exception as e:
    #     logger.error(e)
    print("hello world")
