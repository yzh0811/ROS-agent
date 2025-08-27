import socket

import nacos
import threading
import time

from src import logger
from src.configs.settings import manager, RegisterConfig

service_select = manager.get_service_config("greet-system")
app_host = service_select.app_host
app_port  = service_select.app_port


class NacosHelper:
    def __init__(self, config: RegisterConfig):
        self.client = nacos.NacosClient(
            f"{config.host}:{config.port}",
            namespace=config.namespace
        )
        self.group_name = config.group_name
        self.stop_event = threading.Event()
        self.heartbeat_interval = config.heartbeat_interval
        self.service_name = config.service_name
        self.nacos_host = config.host
        self.nacos_port = config.port
        self.app_port = app_port
        self.app_host = socket.gethostbyname(socket.gethostname())

    def heartbeat_loop(self):
        PORT = self.app_port
        HOST = self.app_host
        while not self.stop_event.is_set():
            try:
                self.client.send_heartbeat(self.service_name, HOST, PORT, self.group_name)
                logger.info("💓 心跳发送成功")
            except Exception as e:
                logger.info(f"⚠️ 心跳发送失败: {e}")
            self.stop_event.wait(self.heartbeat_interval)

    def register_instance(self):
        service_name = self.service_name
        PORT = self.app_port
        HOST = self.app_host
        # SDK 会自动把你传入的 PORT 做 str(port) 转换
        self.client.add_naming_instance(
            service_name,
            HOST,
            PORT,
            group_name=self.group_name
        )
        logger.info(f"✅ 已注册到 Nacos: {service_name} @ {HOST}:{PORT}")

    def deregister(self):
        self.client.remove_naming_instance(self.service_name, self.host, self.port, group_name=self.group_name)
        print("✅ 注销完成")

    def get_service_instance(self, healthy_only=True):
        service_name = self.service_name
        """服务发现：返回一个健康的实例地址（IP, Port）"""
        result = self.client.list_naming_instance(service_name, group_name=self.group_name, healthy_only=healthy_only)
        instances = result.get("hosts", [])
        if not instances:
            raise Exception(f"❌ 未找到服务实例: {service_name}")
        instance = instances[0]
        return instance["ip"], instance["port"]


# register_select = manager.get_register_config("nacos")
# nacos_client = NacosHelper(register_select)
# print("获取第一个服务实例地址")
# print(nacos_client.get_service_instance())
