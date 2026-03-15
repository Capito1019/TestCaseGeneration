import requests
import json
import time

# ============================================================
# 配置区域：请根据“浙大AI接口”提供的实际地址和密钥进行填写
# ============================================================
API_URL = "" 
API_KEY = ""

def get_llm_response(sysPrompt: str, userPrompt: str, max_retries=3) -> str:
    """
    通过 RESTful API 方式调用大模型，并返回生成的对话内容。
    包含 3 次自动重试机制以应对网络超时。
    """
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 构建请求体
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sysPrompt},
            {"role": "user", "content": userPrompt}
        ],
        "temperature": 0.7
    }

    # --- 增加重试逻辑 ---
    for attempt in range(max_retries + 1):
        try:
            # 发送 POST 请求，设置 timeout 为 60 秒
            response = requests.post(
                API_URL, 
                headers=headers, 
                data=json.dumps(data), 
                timeout=60
            )
            
            # 检查响应状态（如果是 5xx 错误也会抛出异常进入 retry）
            response.raise_for_status()
            
            # 解析返回数据
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "ERROR: API 返回数据格式异常，未能找到 choices 字段。"

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # 捕获超时或连接错误
            wait_time = (attempt + 1) * 2  # 等待时间随次数增加：2s, 4s, 6s
            
            if attempt < max_retries:
                print(f"[LLM 警告] 第 {attempt + 1} 次请求超时/异常: {e}。正在重试，等待 {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 达到最大重试次数后的最终报错
                return f"ERROR: 在尝试 {max_retries + 1} 次后调用最终失败。最后一次错误：{str(e)}"

        except requests.exceptions.RequestException as e:
            # 捕获其他 HTTP 错误（如 401 鉴权失败、404 等，通常重试无效）
            return f"ERROR: 请求发生非重试类异常：{str(e)}"
            
        except Exception as e:
            # 捕获解析过程或其他未知错误
            return f"ERROR: 解析过程出错：{str(e)}"

    return "ERROR: 未知逻辑错误导致调用中止。"
