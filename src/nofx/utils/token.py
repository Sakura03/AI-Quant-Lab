import tiktoken


def count_tokens_text(text: str) -> int:
    """
        计算一段纯文本的 token 数量 (system/user/assistant 单条内容)
    """
    # deepseek-chat 使用与 GPT-4 相同的 cl100k_base tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_tokens_messages(messages) -> int:
    """
        计算 chat 格式的消息总 token (包括角色开销)
        DeepSeek 遵循 OpenAI ChatML 风格, 所以格式开销基本一致
        - 每条消息有固定开销: 4
        - assistant 回复前额外有 2 token
    """
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += 4  # ChatML 格式开销
        total += len(enc.encode(msg["content"]))
    total += 2  # assistant 回复开销
    return total
