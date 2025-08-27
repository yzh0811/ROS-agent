from langchain_openai import ChatOpenAI

from src.configs.settings import manager
from src.logger.logger import logger

models_select = manager.get_model_select_config("default-models")
logger.info(f"default chat-model:{models_select.chat_model}")
logger.info(f"default reasoner-model:{models_select.reasoner_model}")
logger.info(f"default reasoner-model:{models_select.embedding_model}")

service_select = manager.get_service_config("greet-system")
logger.info(f"service-host:{service_select.app_host}")
logger.info(f"service-port:{service_select.app_port}")


def get_model(model_name, provider):
    model_config = manager.get_model_config(model_name + provider)
    return ChatOpenAI(
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model_name=model_config.model_key,
        temperature=model_config.temperature
    )

chat_model = get_model(models_select.chat_model, models_select.chat_model_provider)
reasoner_model = get_model(models_select.reasoner_model, models_select.reasoner_model_provider)
