# `/v1/chat/completions` 接口中 `table_info` 动态参数详细生成过程

## 概述

`table_info` 是 `/v1/chat/completions` 对话接口中的一个动态参数，用于在数据库对话场景中向 LLM 提供相关的数据库表结构信息。该参数根据用户查询动态生成，确保 LLM 能够获得与查询最相关的表结构信息。

## 生成流程概览

```
用户请求 → 接口路由 → 创建Chat实例 → 生成table_info → 构建Prompt → LLM推理
```

## 详细生成过程

### 1. 接口入口 (`/v1/chat/completions`)

**文件位置**: `packages/dbgpt-app/src/dbgpt_app/openapi/api_v1/api_v1.py`

**关键代码** (第504-592行):

```python
@router.post("/v1/chat/completions")
async def chat_completions(
    dialogue: ConversationVo = Body(),
    flow_service: FlowService = Depends(get_chat_flow),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    # ... 省略其他代码 ...
    
    # 创建Chat实例
    chat: BaseChat = await get_chat_instance(dialogue)
    
    # 调用stream_generator或no_stream_generator
    return StreamingResponse(
        stream_generator(chat, dialogue.incremental, dialogue.model_name),
        headers=headers,
        media_type="text/plain",
    )
```

### 2. 创建Chat实例 (`get_chat_instance`)

**文件位置**: `packages/dbgpt-app/src/dbgpt_app/openapi/api_v1/api_v1.py` (第447-483行)

**流程**:
1. 验证对话模式 (`chat_mode`)
2. 创建 `ChatParam` 对象，包含：
   - `select_param`: 数据库名称（对于数据库对话场景）
   - `current_user_input`: 用户输入
   - `chat_mode`: 对话场景类型
3. 通过 `ChatFactory` 创建对应的 Chat 实现类

**支持的数据库对话场景**:
- `ChatWithDbExecute`: 数据库自动执行对话
- `ChatWithDbQA`: 数据库专业问答
- `ChatDashboard`: 数据看板生成

### 3. Chat实例的 `generate_input_values` 方法

这是 `table_info` 生成的核心方法，不同场景的实现略有不同：

#### 3.1 ChatWithDbAutoExecute (数据库自动执行)

**文件位置**: `packages/dbgpt-app/src/dbgpt_app/scene/chat_db/auto_execute/chat.py` (第49-88行)

**生成流程**:

```python
async def generate_input_values(self) -> Dict:
    user_input = self.current_user_input.last_text
    client = DBSummaryClient(system_app=self.system_app)
    
    try:
        # 方式1: 使用向量检索获取相关表信息（优先）
        table_infos = await blocking_func_to_async(
            self._executor,
            client.get_db_summary,
            self.db_name,           # 数据库名称
            user_input,             # 用户查询
            self.curr_config.schema_retrieve_top_k,  # Top-K值
        )
    except Exception as e:
        # 方式2: 降级方案 - 获取所有表的简单信息
        logger.error(f"Retrieved table info error: {str(e)}")
        table_infos = await blocking_func_to_async(
            self._executor, 
            self.database.table_simple_info
        )
        # 如果表信息过长，进行截断
        if len(table_infos) > self.curr_config.schema_max_tokens:
            table_infos = table_infos[: self.curr_config.schema_max_tokens]
    
    return {
        "db_name": self.db_name,
        "user_input": user_input,
        "table_info": table_infos,  # 这里就是生成的table_info
        # ... 其他参数
    }
```

#### 3.2 ChatWithDbQA (数据库专业问答)

**文件位置**: `packages/dbgpt-app/src/dbgpt_app/scene/chat_db/professional_qa/chat.py` (第52-85行)

生成流程与 `ChatWithDbAutoExecute` 类似，但返回的参数略有不同。

### 4. DBSummaryClient.get_db_summary 方法

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第63-79行)

**功能**: 基于用户查询，从向量存储中检索最相关的表结构信息

**实现**:

```python
def get_db_summary(self, dbname, query, topk):
    """Get user query related tables info."""
    from dbgpt_ext.rag.retriever.db_schema import DBSchemaRetriever
    
    # 1. 获取向量存储连接器
    table_vector_connector, field_vector_connector = (
        self._get_vector_connector_by_db(dbname)
    )
    
    # 2. 创建DBSchemaRetriever
    retriever = DBSchemaRetriever(
        top_k=topk,
        table_vector_store_connector=table_vector_connector,
        field_vector_store_connector=field_vector_connector,
        separator="--table-field-separator--",
    )
    
    # 3. 执行检索
    table_docs = retriever.retrieve(query)
    
    # 4. 提取内容
    ans = [d.content for d in table_docs]
    return ans
```

**向量存储命名规则**:
- 表向量存储: `{dbname}_profile`
- 字段向量存储: `{dbname}_profile_field`

### 5. DBSchemaRetriever.retrieve 方法

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/retriever/db_schema.py` (第121-231行)

**检索流程**:

#### 5.1 相似度搜索 (`_similarity_search`)

```python
def _similarity_search(self, query, filters=None) -> List[Chunk]:
    # 1. 从表向量存储中检索Top-K相关表
    table_chunks = self._table_vector_store_connector.similar_search_with_scores(
        query, self._top_k, 0, filters
    )
    
    # 2. 分类处理
    not_sep_chunks = [
        chunk for chunk in table_chunks 
        if not chunk.metadata.get("separated")
    ]
    separated_chunks = [
        chunk for chunk in table_chunks 
        if chunk.metadata.get("separated")
    ]
    
    # 3. 对于分离的表（字段信息单独存储），需要额外检索字段信息
    if separated_chunks:
        tasks = [
            lambda c=chunk: self._retrieve_field(c, query) 
            for chunk in separated_chunks
        ]
        separated_result = run_tasks(tasks, concurrency_limit=3)
        return not_sep_chunks + separated_result
    
    return [self._deserialize_table_chunk(chunk) for chunk in not_sep_chunks]
```

#### 5.2 字段检索 (`_retrieve_field`)

对于字段信息单独存储的表，需要从字段向量存储中检索相关字段：

```python
def _retrieve_field(self, table_chunk: Chunk, query) -> Chunk:
    # 1. 构建元数据过滤器
    metadata = table_chunk.metadata
    metadata["part"] = "field"
    filters = [MetadataFilter(key=k, value=v) for k, v in metadata.items()]
    
    # 2. 从字段向量存储中检索
    field_chunks = self._field_vector_store_connector.similar_search_with_scores(
        query, self._top_k, 0, MetadataFilters(filters=filters)
    )
    
    # 3. 合并表信息和字段信息
    field_contents = [chunk.content.strip() for chunk in field_chunks]
    table_chunk.content += (
        "\n" + self._separator + "\n" + 
        self._column_separator.join(field_contents)
    )
    
    return self._deserialize_table_chunk(table_chunk)
```

#### 5.3 反序列化表块 (`_deserialize_table_chunk`)

将检索到的表信息转换为 CREATE TABLE 语句格式：

```python
def _deserialize_table_chunk(self, chunk: Chunk) -> Chunk:
    # 解析表信息
    parts = chunk.content.split(self._separator)
    table_part, field_part = parts[0].strip(), parts[1].strip()
    table_detail = _parse_table_detail(table_part)
    
    # 构建CREATE TABLE语句
    create_statement = f"CREATE TABLE `{table_name}`\r\n(\r\n    "
    create_statement += field_part
    create_statement += "\r\n)"
    if table_comment:
        create_statement += f' COMMENT "{table_comment}"\r\n'
    if index_keys:
        create_statement += f"Index keys: {index_keys}"
    
    chunk.content = create_statement
    return chunk
```

### 6. 降级方案: table_simple_info

当向量检索失败时，系统会降级使用 `table_simple_info` 方法获取所有表的简单信息。

**文件位置**: `packages/dbgpt-core/src/dbgpt/datasource/rdbms/base.py` (第305-315行)

**实现** (MySQL示例):

```python
def table_simple_info(self):
    """Return table simple info."""
    _sql = f"""
        select concat(table_name, "(" , group_concat(column_name), ")")
        as schema_info from information_schema.COLUMNS where
        table_schema="{self.get_current_db_name()}" 
        group by TABLE_NAME;
    """
    with self.session_scope() as session:
        cursor = session.execute(text(_sql))
        results = cursor.fetchall()
        return results
```

**不同数据库的实现**:
- **SQLite**: 使用 `PRAGMA table_info` 命令
- **PostgreSQL**: 查询 `pg_catalog` 系统表
- **MySQL**: 查询 `information_schema.COLUMNS`

### 7. 完整的 get_table_info 方法（备用方案）

**文件位置**: `packages/dbgpt-core/src/dbgpt/datasource/rdbms/base.py` (第322-368行)

当需要获取完整的表信息（包括索引、示例数据）时，使用此方法：

```python
def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
    """Get information about specified tables."""
    # 1. 获取表名列表
    all_table_names = self.get_usable_table_names()
    if table_names is not None:
        all_table_names = table_names
    
    # 2. 获取元数据表对象
    meta_tables = [
        tbl for tbl in self._metadata.sorted_tables
        if tbl.name in set(all_table_names)
    ]
    
    # 3. 为每个表生成CREATE TABLE语句
    tables = []
    for table in meta_tables:
        # 使用SQLAlchemy生成CREATE TABLE语句
        create_table = str(CreateTable(table).compile(self._engine))
        table_info = f"{create_table.rstrip()}"
        
        # 可选：添加索引信息
        if self._indexes_in_table_info:
            table_info += f"\n{self._get_table_indexes(table)}\n"
        
        # 可选：添加示例数据
        if self._sample_rows_in_table_info:
            table_info += f"\n{self._get_sample_rows(table)}\n"
        
        tables.append(table_info)
    
    # 4. 合并所有表信息
    final_str = "\n\n".join(tables)
    return final_str
```

## 数据流转图

```
用户查询
    ↓
/v1/chat/completions 接口
    ↓
get_chat_instance() → ChatWithDbAutoExecute/ChatWithDbQA
    ↓
generate_input_values()
    ↓
DBSummaryClient.get_db_summary()
    ↓
DBSchemaRetriever.retrieve()
    ↓
├─→ 表向量存储检索 (table_vector_store)
│   └─→ 相似度搜索 Top-K 表
│
└─→ 字段向量存储检索 (field_vector_store) [可选]
    └─→ 为分离的表检索相关字段
    ↓
反序列化为 CREATE TABLE 语句
    ↓
table_info (List[str])
    ↓
构建 Prompt → LLM 推理
```

## 关键配置参数

### 1. schema_retrieve_top_k
- **位置**: `ChatWithDBExecuteConfig`
- **作用**: 向量检索时返回的Top-K表数量
- **默认值**: 通常为 4-10

### 2. schema_max_tokens
- **位置**: `ChatWithDBExecuteConfig`
- **作用**: table_info 的最大token数限制
- **用途**: 防止表信息过长导致超出模型上下文窗口

### 3. sample_rows_in_table_info
- **位置**: `RDBMSConnector` 初始化参数
- **作用**: 是否在表信息中包含示例数据行数
- **默认值**: 3

### 4. indexes_in_table_info
- **位置**: `RDBMSConnector` 初始化参数
- **作用**: 是否在表信息中包含索引信息
- **默认值**: False

## 向量存储初始化

在首次使用某个数据库时，需要初始化向量存储：

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第96-130行)

**流程**:
1. 检查向量存储是否已存在
2. 如果不存在，使用 `DBSchemaAssembler` 加载数据库连接
3. 将表结构和字段信息切分为chunks
4. 生成embeddings并存储到向量数据库
5. 持久化向量存储

## 总结

`table_info` 的生成采用了**智能检索 + 降级方案**的策略：

1. **优先方案**: 使用向量相似度检索，根据用户查询动态获取最相关的表结构信息
2. **降级方案**: 如果向量检索失败，获取所有表的简单信息并截断
3. **完整方案**: 在需要时，可以获取包含索引、示例数据的完整表信息

这种设计既保证了效率（只返回相关表），又保证了可靠性（有降级方案），同时还能根据需求灵活调整信息详细程度。

