# -*- coding: utf-8 -*-
"""
阿里云 DashScope SDK Python 版 Embedding 使用示例
"""

import dashscope
from dashscope import TextEmbedding
from http import HTTPStatus


# 设置全局 API Key 和 API URL
dashscope.api_key = 'sk-3159e8ae1549433e8815cd3eafd3a4f8'
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


def example_1_basic_usage():
    """基础用法：单个文本嵌入"""
    print("=" * 50)
    print("示例1: 基础用法 - 单个文本嵌入")
    print("=" * 50)
    
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v4,
        input='风急天高猿啸哀'
    )
    
    if resp.status_code == HTTPStatus.OK:
        # 获取嵌入向量
        embedding = resp['output']['embeddings'][0]['embedding']
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")
    else:
        print(f"错误: {resp.message}")


def example_2_batch_embedding():
    """批量文本嵌入（最多支持10条）"""
    print("\n" + "=" * 50)
    print("示例2: 批量文本嵌入")
    print("=" * 50)
    
    texts = [
        '风急天高猿啸哀',
        '渚清沙白鸟飞回',
        '无边落木萧萧下',
        '不尽长江滚滚来'
    ]
    
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v4,
        input=texts
    )
    
    if resp.status_code == HTTPStatus.OK:
        embeddings = resp['output']['embeddings']
        print(f"成功嵌入 {len(embeddings)} 条文本")
        for i, emb in enumerate(embeddings):
            print(f"文本 {i+1} 向量维度: {len(emb['embedding'])}")
    else:
        print(f"错误: {resp.message}")




if __name__ == '__main__':
    print("DashScope TextEmbedding 使用示例\n")
    
    # 取消注释以运行示例
    # example_1_basic_usage()
    example_2_batch_embedding()

