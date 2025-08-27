###################################
# 使用官方 Python 轻量级基础镜像
###################################
FROM python:3.12-slim

###################################
# 设置工作目录
###################################
WORKDIR /app

## 先复制 wheel 包
#COPY torch_wheels /wheels
#
## 安装 PyTorch wheel 包（避免在线下载）
#RUN pip install /wheels/torch-2.2.0+cpu-cp311-cp311-win_amd64.whl

###################################
# 安装系统依赖（按需添加）
###################################
# 如果你的 Python 包需要编译（如 psycopg2 等），需安装 gcc
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends gcc python3-dev && \
#    rm -rf /var/lib/apt/lists/*

###################################
# 复制依赖文件并安装
###################################

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

###################################
# 复制项目代码
###################################
COPY . .

###################################
# 暴露服务端口
###################################
EXPOSE 7860

###################################
# 启动 Uvicorn 服务
###################################
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]

#构建服务
#docker build -t project-fastapi-service-greet-system .

#导出镜像
#docker save -o project-fastapi-service-greet-system.tar project-fastapi-service-greet-system

#导出到特定文件夹
#docker save -o "C:\Users\Administrator\Desktop\python-jar\project-fastapi-service-greet-system.tar" project-fastapi-service-greet-system

#将镜像解压并导入到服务器的Docker中
#docker load -i project-fastapi-service-greet-system.tar

#运行容器
#docker run -d -p 7860:7860 --name project-fastapi-service-greet-system-1 project-fastapi-service-greet-system

#运行以下命令安装 Nacos 的 Python SDK
#pip install nacos-sdk-python

#停止并删除容器
#docker stop greet-app
#docker rm greet-app
#删除镜像
#docker rmi my-fastapi-greet

#docker-compose启动，已有镜像
#docker-compose up -d fastapi-service-greet-system

#如果docker-compose有其他名字，用另外的命令启动
#docker-compose -f my-compose-file.yml up -d fastapi-service-greet-system





