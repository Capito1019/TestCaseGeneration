import os

#OSS
OSS_ACCESS_KEY_ID = ""
OSS_ACCESS_KEY_SECRET = ""
OSS_ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"  # 根据你的Bucket地域改
OSS_BUCKET_NAME = "doc-to-test"
OSS_PUBLIC_DOMAIN = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}"
OSS_OBJECT_PREFIX = "08afb457-f0f6-45f4-a589-fd52e94d9cf2/"

#MinerU
API_TOKEN = os.getenv("MINERU_API_TOKEN", "")
MINERU_BASE_URL = "https://mineru.net/api/v4"

#openai
API_KEY    = os.getenv("OPENAI_API_KEY", "")
BASE_URL   = os.getenv("OPENAI_BASE_URL", "")
MODEL      = os.getenv("MODEL", "gpt-4o-ca")  # 支持图片输入的多模态模型