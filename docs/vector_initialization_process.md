# 数据库向量初始化详细逻辑

## 概述

数据库向量初始化是将数据库表结构信息转换为向量嵌入并存储到向量数据库的过程。这个过程使得系统能够通过语义相似度检索快速找到与用户查询相关的表结构信息。

## 初始化流程图

```
触发初始化
    ↓
创建 DBSummaryClient
    ↓
创建 RdbmsSummary/GdbmsSummary
    ↓
获取向量存储连接器
    ↓
检查向量存储是否存在
    ↓
[不存在] → 创建 DBSchemaAssembler
    ↓
DatasourceKnowledge 提取表结构
    ↓
切分为 Chunks
    ↓
生成 Embeddings
    ↓
持久化到向量存储
```

## 详细步骤

### 1. 触发入口

#### 1.1 手动触发 - `db_summary_embedding`

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第48-61行)

**调用场景**:
- 用户通过 API 手动刷新数据库向量
- 新增数据库后需要初始化向量

**代码**:
```python
def db_summary_embedding(self, dbname, db_type):
    """Put db profile and table profile summary into vector store."""
    try:
        # 1. 创建数据库摘要客户端
        db_summary_client = self.create_summary_client(dbname, db_type)
        
        # 2. 初始化数据库配置文件
        self.init_db_profile(db_summary_client, dbname)
        
        logger.info("db summary embedding success")
    except Exception as e:
        logger.warning(f"{dbname}, {db_type} summary error!{str(e)}")
        raise
```

#### 1.2 批量初始化 - `init_db_summary`

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第81-94行)

**调用场景**:
- 系统启动时自动初始化所有数据库
- 后台线程异步执行

**代码**:
```python
def init_db_summary(self):
    """Initialize db summary profile."""
    local_db_manager = ConnectorManager.get_instance(self.system_app)
    dbs = local_db_manager.get_db_list()
    
    # 遍历所有数据库，逐个初始化
    for item in dbs:
        try:
            self.db_summary_embedding(item["db_name"], item["db_type"])
        except Exception as e:
            logger.warning(
                f"{item['db_name']}, {item['db_type']} summary error!{str(e)}"
            )
```

### 2. 创建数据库摘要客户端

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第145-157行)

**功能**: 根据数据库类型创建对应的摘要客户端

**代码**:
```python
@staticmethod
def create_summary_client(dbname: str, db_type: str):
    """Create a summary client based on the database type."""
    if "graph" in db_type:
        # 图数据库
        return GdbmsSummary(dbname, db_type)
    else:
        # 关系型数据库
        return RdbmsSummary(dbname, db_type)
```

**RdbmsSummary 初始化过程**:

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/summary/rdbms_db_summary.py` (第53-80行)

```python
def __init__(self, name: str, type: str, manager: Optional["ConnectorManager"] = None):
    """Create a new RdbmsSummary."""
    self.name = name
    self.type = type
    self.summary_template = "{table_name}({columns})"
    
    # 获取数据库连接器
    db_manager = manager or CFG.local_db_manager
    self.db = db_manager.get_connector(name)
    
    # 获取数据库元数据信息
    self.metadata = """user info :{users}, grant info:{grant}, 
                       charset:{charset}, collation:{collation}""".format(
        users=self.db.get_users(),
        grant=self.db.get_grants(),
        charset=self.db.get_charset(),
        collation=self.db.get_collation(),
    )
    
    # 获取所有表名并生成表摘要
    tables = self.db.get_table_names()
    self.table_info_summaries = [
        self.get_table_summary(table_name) for table_name in tables
    ]
```

### 3. 获取向量存储连接器

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第159-171行)

**功能**: 为数据库创建两个向量存储连接器

**代码**:
```python
def _get_vector_connector_by_db(self, dbname) -> Tuple[VectorStoreBase, VectorStoreBase]:
    """获取表向量存储和字段向量存储连接器"""
    # 表向量存储名称: {dbname}_profile
    vector_store_name = dbname + "_profile"
    storage_manager = StorageManager.get_instance(self.system_app)
    table_vector_store = storage_manager.create_vector_store(
        index_name=vector_store_name
    )
    
    # 字段向量存储名称: {dbname}_profile_field
    field_vector_store_name = dbname + "_profile_field"
    field_vector_store = storage_manager.create_vector_store(
        index_name=field_vector_store_name
    )
    
    return table_vector_store, field_vector_store
```

**向量存储命名规则**:
- **表向量存储**: `{数据库名}_profile` - 存储表的基本信息和字段信息
- **字段向量存储**: `{数据库名}_profile_field` - 当表字段过多时，单独存储字段信息

### 4. 检查向量存储是否存在

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第96-130行)

**逻辑**: 如果向量存储已存在，跳过初始化；否则执行初始化

```python
def init_db_profile(self, db_summary_client, dbname):
    """Initialize db summary profile."""
    vector_store_name = dbname + "_profile"
    
    # 获取向量存储连接器
    table_vector_connector, field_vector_connector = (
        self._get_vector_connector_by_db(dbname)
    )
    
    # 检查向量存储是否已存在
    if not table_vector_connector.vector_name_exists():
        # 执行初始化流程
        # ... (见下一节)
    else:
        logger.info(f"Vector store name {vector_store_name} exist")
```

### 5. 创建 DBSchemaAssembler

**文件位置**: `packages/dbgpt-serve/src/dbgpt_serve/datasource/service/db_summary_client.py` (第108-124行)

**代码**:
```python
if not table_vector_connector.vector_name_exists():
    from dbgpt_ext.rag.assembler.db_schema import DBSchemaAssembler
    from dbgpt_ext.rag.summary.rdbms_db_summary import _DEFAULT_COLUMN_SEPARATOR
    
    # 配置文本切分参数
    chunk_parameters = ChunkParameters(
        text_splitter=RDBTextSplitter(
            column_separator=_DEFAULT_COLUMN_SEPARATOR,  # ",\r\n    "
            separator="--table-field-separator--",  # 表信息和字段信息的分隔符
        )
    )
    
    # 创建 DBSchemaAssembler
    db_assembler = DBSchemaAssembler.load_from_connection(
        connector=db_summary_client.db,  # 数据库连接器
        table_vector_store_connector=table_vector_connector,  # 表向量存储
        field_vector_store_connector=field_vector_connector,  # 字段向量存储
        chunk_parameters=chunk_parameters,  # 切分参数
        max_seq_length=self.app_config.service.web.embedding_model_max_seq_len,  # 最大序列长度
    )
    
    # 如果生成了chunks，则持久化
    if len(db_assembler.get_chunks()) > 0:
        db_assembler.persist()
```

### 6. DBSchemaAssembler 初始化过程

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/assembler/db_schema.py` (第37-74行)

**初始化流程**:

```python
def __init__(
    self,
    connector: BaseConnector,
    table_vector_store_connector: VectorStoreBase,
    field_vector_store_connector: Optional[VectorStoreBase] = None,
    chunk_parameters: Optional[ChunkParameters] = None,
    embedding_model: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
    max_seq_length: int = 512,
    **kwargs: Any,
) -> None:
    """Initialize with Embedding Assembler arguments."""
    self._connector = connector
    self._table_vector_store_connector = table_vector_store_connector
    self._field_vector_store_connector = field_vector_store_connector
    
    # 初始化嵌入模型
    if self._embedding_model and not embeddings:
        embeddings = DefaultEmbeddingFactory(
            default_model_name=self._embedding_model
        ).create(self._embedding_model)
    
    # 创建 DatasourceKnowledge 对象
    knowledge = DatasourceKnowledge(
        connector, 
        model_dimension=max_seq_length
    )
    
    # 调用父类初始化，会自动调用 load_knowledge
    super().__init__(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        **kwargs,
    )
```

**BaseAssembler 的初始化** (第18-53行):

```python
def __init__(
    self,
    knowledge: Knowledge,
    chunk_parameters: Optional[ChunkParameters] = None,
    extractor: Optional[ExtractorBase] = None,
    **kwargs: Any,
) -> None:
    self._knowledge = knowledge
    self._chunk_parameters = chunk_parameters or ChunkParameters()
    self._chunk_manager = ChunkManager(
        knowledge=self._knowledge, 
        chunk_parameter=self._chunk_parameters
    )
    self._chunks: List[Chunk] = []
    
    # 自动加载知识并切分为chunks
    self.load_knowledge(self._knowledge)
```

**load_knowledge 方法** (第55-62行):

```python
def load_knowledge(self, knowledge: Knowledge) -> None:
    """Load knowledge Pipeline."""
    if not knowledge:
        raise ValueError("knowledge must be provided.")
    
    # 1. 从数据库加载文档
    with root_tracer.start_span("BaseAssembler.knowledge.load"):
        documents = knowledge.load()
    
    # 2. 将文档切分为chunks
    with root_tracer.start_span("BaseAssembler.chunk_manager.split"):
        self._chunks = self._chunk_manager.split(documents)
```

### 7. DatasourceKnowledge 提取表结构信息

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/knowledge/datasource.py` (第54-71行)

**功能**: 从数据库连接中提取所有表的结构信息并转换为 Document 对象

**代码**:
```python
def _load(self) -> List[Document]:
    """Load datasource document from data_loader."""
    docs = []
    
    # 解析数据库摘要（包含元数据）
    db_summary_with_metadata = _parse_db_summary_with_metadata(
        self._connector,
        self._summary_template,
        self._separator,
        column_separator=self._column_separator,
        model_dimension=self._model_dimension,
    )
    
    # 为每个表创建 Document 对象
    for summary, table_metadata in db_summary_with_metadata:
        metadata = {"source": "database"}
        
        if self._metadata:
            metadata.update(self._metadata)
        
        # 合并表元数据
        table_metadata.update(metadata)
        
        # 创建 Document
        docs.append(Document(content=summary, metadata=table_metadata))
    
    return docs
```

### 8. 解析表摘要信息

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/summary/rdbms_db_summary.py` (第113-141行)

**功能**: 遍历所有表，解析每个表的结构信息

**代码**:
```python
def _parse_db_summary_with_metadata(
    conn: BaseConnector,
    summary_template: str = _DEFAULT_SUMMARY_TEMPLATE,
    separator: str = "--table-field-separator--",
    column_separator: str = _DEFAULT_COLUMN_SEPARATOR,
    model_dimension: int = 512,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get db summary for database."""
    # 获取所有表名
    tables = conn.get_table_names()
    
    # 为每个表生成摘要和元数据
    table_info_summaries = [
        _parse_table_summary_with_metadata(
            conn,
            summary_template,
            separator,
            table_name,
            model_dimension,
            column_separator=column_separator,
        )
        for table_name in tables
    ]
    return table_info_summaries
```

### 9. 解析单个表摘要

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/summary/rdbms_db_summary.py` (第181-258行)

**功能**: 解析单个表的结构信息，包括表名、注释、索引、字段等

**详细流程**:

```python
def _parse_table_summary_with_metadata(
    conn: BaseConnector,
    summary_template: str,
    separator,
    table_name: str,
    model_dimension=512,
    column_separator: str = _DEFAULT_COLUMN_SEPARATOR,
    db_summary_version: str = "v1.0",
) -> Tuple[str, Dict[str, Any]]:
    """Get table summary for table."""
    
    # 1. 初始化元数据
    metadata = {
        "table_name": table_name,
        "separated": 0,  # 是否分离存储（字段过多时）
        "db_summary_version": db_summary_version,
    }
    
    # 2. 获取所有列信息
    columns = []
    for column in conn.get_columns(table_name):
        col_name = column["name"]
        col_type = str(column["type"]) if "type" in column else None
        col_comment = column.get("comment")
        
        # 构建列定义字符串
        column_def = f'"{col_name}" {col_type.upper()}'
        if col_comment:
            column_def += f' COMMENT "{col_comment}"'
        columns.append(column_def)
    
    metadata.update({"field_num": len(columns)})
    
    # 3. 如果字段过多，需要切分
    separated_columns = _split_columns_str(
        columns, 
        model_dimension=model_dimension, 
        column_separator=column_separator
    )
    
    # 如果字段被切分为多段，标记为分离存储
    if len(separated_columns) > 1:
        metadata["separated"] = 1
    
    column_str = column_separator.join(separated_columns)
    
    # 4. 获取索引信息
    index_keys = []
    raw_indexes = conn.get_indexes(table_name)
    for index in raw_indexes:
        if isinstance(index, tuple):
            index_name, index_creation_command = index
            # 使用正则表达式提取列名
            matched_columns = re.findall(r"\(([^)]+)\)", index_creation_command)
            if matched_columns:
                key_str = ", ".join(matched_columns)
                index_keys.append(f"{index_name}(`{key_str}`) ")
        else:
            key_str = ", ".join(index["column_names"])
            index_keys.append(f"{index['name']}(`{key_str}`) ")
    
    # 5. 获取表注释
    table_comment = ""
    try:
        comment = conn.get_table_comment(table_name)
        table_comment = comment.get("text")
    except Exception:
        pass
    
    # 6. 构建表摘要字符串
    index_key_str = ", ".join(index_keys)
    table_str = summary_template.format(
        table_name=table_name, 
        table_comment=table_comment, 
        index_keys=index_key_str
    )
    
    # 7. 添加字段信息（用分隔符分开）
    table_str += f"\n{separator}\n{column_str}"
    
    return table_str, metadata
```

**表摘要格式示例**:
```
table_name: users
table_comment: 用户信息表
index_keys: PRIMARY KEY(`id`), INDEX idx_email(`email`)

--table-field-separator--

"id" INTEGER COMMENT "用户ID",
"name" VARCHAR(50) COMMENT "用户名",
"email" VARCHAR(100) COMMENT "邮箱"
```

### 10. 字段字符串切分

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/summary/rdbms_db_summary.py` (第144-178行)

**功能**: 当表的字段信息过长时，按照 `model_dimension` 阈值切分为多个片段

**代码**:
```python
def _split_columns_str(
    columns: List[str], 
    model_dimension: int, 
    column_separator: str = ",\r\n    "
):
    """Split columns str.
    
    Args:
        columns: 字段列表
        model_dimension: 切分阈值（字符长度）
    """
    result = []
    current_string = ""
    current_length = 0
    
    for element_str in columns:
        element_length = len(element_str)
        
        # 如果添加当前元素会超过阈值，保存当前字符串并重置
        if current_length + element_length > model_dimension:
            result.append(current_string.strip())
            current_string = element_str
            current_length = element_length
        else:
            # 否则追加到当前字符串
            if current_string:
                current_string += column_separator + element_str
            else:
                current_string = element_str
            current_length += element_length + 1
    
    # 处理最后一个字符串段
    if current_string:
        result.append(current_string.strip())
    
    return result
```

**切分逻辑**:
- 如果字段总长度 <= `model_dimension`: 不切分，`separated = 0`
- 如果字段总长度 > `model_dimension`: 切分为多段，`separated = 1`

### 11. ChunkManager 切分文档

**功能**: 将 Document 对象切分为 Chunk 对象

**切分策略**: 
- 对于数据库表结构，使用 `CHUNK_BY_PAGE` 策略
- 每个表对应一个 Chunk（如果字段未分离）或多个 Chunks（如果字段已分离）

**Chunk 结构**:
```python
Chunk(
    content="表摘要字符串",
    metadata={
        "table_name": "users",
        "separated": 0,  # 或 1
        "field_num": 10,
        "db_summary_version": "v1.0",
        "source": "database",
        "part": "table"  # 或 "field"（如果分离）
    }
)
```

### 12. 持久化到向量存储

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/rag/assembler/db_schema.py` (第116-135行)

**功能**: 将 Chunks 分类并持久化到对应的向量存储

**代码**:
```python
def persist(self, **kwargs: Any) -> List[str]:
    """Persist chunks into vector store."""
    table_chunks, field_chunks = [], []
    
    # 1. 分类 Chunks
    for chunk in self._chunks:
        metadata = chunk.metadata
        if metadata.get("separated"):
            # 分离存储的表
            if metadata.get("part") == "table":
                table_chunks.append(chunk)
            else:
                field_chunks.append(chunk)
        else:
            # 未分离的表（表信息和字段信息在一起）
            table_chunks.append(chunk)
    
    # 2. 持久化字段 Chunks（如果存在）
    if self._field_vector_store_connector and field_chunks:
        self._field_vector_store_connector.load_document_with_limit(field_chunks)
    
    # 3. 持久化表 Chunks
    return self._table_vector_store_connector.load_document_with_limit(table_chunks)
```

**持久化过程** (`load_document_with_limit`):
1. 为每个 Chunk 生成 Embedding 向量
2. 将 Chunk 内容、向量、元数据存储到向量数据库
3. 返回存储的 Chunk IDs

## 数据流转示例

### 示例：初始化 `test_db` 数据库

**输入**:
- 数据库名: `test_db`
- 表: `users`, `orders`

**处理流程**:

1. **创建向量存储连接器**:
   - 表向量存储: `test_db_profile`
   - 字段向量存储: `test_db_profile_field`

2. **提取表结构**:
   ```
   users 表:
   - 字段: id, name, email (3个字段，未超过阈值)
   - 索引: PRIMARY KEY(id)
   - 注释: 用户信息表
   
   orders 表:
   - 字段: id, user_id, product_id, ..., (50个字段，超过阈值)
   - 索引: PRIMARY KEY(id), INDEX idx_user_id(user_id)
   - 注释: 订单表
   ```

3. **生成 Chunks**:
   ```
   Chunk 1 (users):
   - content: "table_name: users\ntable_comment: 用户信息表\n..."
   - metadata: {"table_name": "users", "separated": 0, "field_num": 3}
   
   Chunk 2 (orders - 表部分):
   - content: "table_name: orders\ntable_comment: 订单表\n..."
   - metadata: {"table_name": "orders", "separated": 1, "part": "table"}
   
   Chunk 3 (orders - 字段部分1):
   - content: "字段定义1..."
   - metadata: {"table_name": "orders", "separated": 1, "part": "field"}
   
   Chunk 4 (orders - 字段部分2):
   - content: "字段定义2..."
   - metadata: {"table_name": "orders", "separated": 1, "part": "field"}
   ```

4. **生成 Embeddings**:
   - 为每个 Chunk 的内容生成向量表示

5. **持久化**:
   - Chunk 1, 2 → `test_db_profile` (表向量存储)
   - Chunk 3, 4 → `test_db_profile_field` (字段向量存储)

## 关键配置参数

### 1. max_seq_length (model_dimension)
- **位置**: `app_config.service.web.embedding_model_max_seq_len`
- **作用**: 字段字符串切分的阈值
- **默认值**: 512
- **影响**: 超过此长度的字段列表会被切分为多段

### 2. column_separator
- **位置**: `_DEFAULT_COLUMN_SEPARATOR`
- **值**: `",\r\n    "`
- **作用**: 字段之间的分隔符

### 3. separator
- **位置**: 固定值
- **值**: `"--table-field-separator--"`
- **作用**: 表基本信息和字段信息之间的分隔符

### 4. db_summary_version
- **位置**: 固定值
- **值**: `"v1.0"`
- **作用**: 数据库摘要版本号，用于兼容性控制

## 向量存储结构

### 表向量存储 (`{dbname}_profile`)

存储内容:
- 表的基本信息（表名、注释、索引）
- 字段信息（如果字段未分离）

元数据:
```json
{
  "table_name": "users",
  "separated": 0,
  "field_num": 3,
  "db_summary_version": "v1.0",
  "source": "database"
}
```

### 字段向量存储 (`{dbname}_profile_field`)

存储内容:
- 仅存储字段定义信息（当 `separated = 1` 时）

元数据:
```json
{
  "table_name": "orders",
  "separated": 1,
  "part": "field",
  "field_num": 50,
  "db_summary_version": "v1.0",
  "source": "database"
}
```

## 检索时的使用

当用户查询时，检索流程如下:

1. **表向量存储检索**: 根据查询找到相关的表
2. **字段向量存储检索**: 如果表是分离存储的，根据表名和查询找到相关字段
3. **合并结果**: 将表信息和字段信息合并为完整的 CREATE TABLE 语句

## 向量数据存储位置

### 1. 默认存储类型

系统默认使用 **Chroma** 作为向量存储，但支持多种向量存储类型：
- **Chroma** (默认): 本地文件系统存储
- **Milvus**: 远程向量数据库
- **Weaviate**: 远程向量数据库
- **OceanBase**: 关系型数据库（支持向量）

### 2. Chroma 存储位置

**文件位置**: `packages/dbgpt-ext/src/dbgpt_ext/storage/vector_store/chroma_store.py` (第116-120行)

**路径确定逻辑**:

```python
# 1. 从配置中获取 persist_path
chroma_path = chroma_vector_config.get(
    "persist_path", 
    os.path.join(PILOT_PATH, "data")  # 默认值
)

# 2. 解析为绝对路径
self.persist_dir = os.path.join(
    resolve_root_path(chroma_path) + "/chromadb"
)
```

**路径优先级**:
1. **配置文件**: `[rag.storage.vector].persist_path` (在 TOML 配置文件中)
2. **环境变量**: `CHROMA_PERSIST_PATH`
3. **默认路径**: `{ROOT_PATH}/pilot/data/chromadb`

**示例配置** (`configs/dbgpt-local-vllm.toml`):
```toml
[rag.storage]
[rag.storage.vector]
type = "chroma"
persist_path = "pilot/data"  # 相对路径，会解析为 {ROOT_PATH}/pilot/data/chromadb
```

**最终存储路径**:
- 如果配置 `persist_path = "pilot/data"`: `{项目根目录}/pilot/data/chromadb`
- 如果配置 `persist_path = "/data/vectors"`: `/data/vectors/chromadb`
- 如果未配置: `{项目根目录}/pilot/data/chromadb`

### 3. Chroma 存储结构

Chroma 使用 **PersistentClient** 将数据持久化到本地文件系统。

**存储目录结构**:
```
{persist_dir}/
├── chroma.sqlite3          # Chroma 元数据数据库
├── {collection_name}/      # 每个集合一个目录
│   ├── data_level0.bin    # 数据文件
│   ├── header.bin          # 头部信息
│   └── link_lists.bin      # 链接列表
└── ...
```

**数据库向量存储集合**:
- **表向量集合**: `{dbname}_profile`
- **字段向量集合**: `{dbname}_profile_field`

**示例** (数据库名: `test_db`):
```
pilot/data/chromadb/
├── chroma.sqlite3
├── test_db_profile/        # 表向量存储
│   └── ...
└── test_db_profile_field/  # 字段向量存储
    └── ...
```

### 4. 其他向量存储类型

#### 4.1 Milvus

**存储位置**: 远程 Milvus 服务器

**配置**:
```toml
[rag.storage.vector]
type = "milvus"
uri = "127.0.0.1"      # Milvus 服务器地址
port = "19530"          # Milvus 端口
user = "root"           # 用户名（可选）
password = "password"    # 密码（可选）
```

**环境变量**:
- `MILVUS_URL`: Milvus 服务器地址
- `MILVUS_PORT`: Milvus 端口
- `MILVUS_USERNAME`: 用户名
- `MILVUS_PASSWORD`: 密码

**存储位置**: Milvus 服务器内部存储（通常是配置的数据目录）

#### 4.2 Weaviate

**存储位置**: 远程 Weaviate 服务器

**配置**:
```toml
[rag.storage.vector]
type = "weaviate"
weaviate_url = "https://your-weaviate-instance.weaviate.network"
persist_path = "/path/to/persist"  # 本地持久化路径（可选）
```

**环境变量**:
- `WEAVIATE_URL`: Weaviate 服务器 URL
- `WEAVIATE_PERSIST_PATH`: 本地持久化路径

**存储位置**: 
- 如果使用 Weaviate Cloud: 云端存储
- 如果使用本地 Weaviate: 服务器配置的数据目录

### 5. 查看向量数据存储位置

#### 5.1 查看配置

检查配置文件中的 `persist_path` 设置：

```bash
# 查看配置文件
cat configs/dbgpt-local-vllm.toml | grep -A 3 "\[rag.storage.vector\]"
```

#### 5.2 查看实际存储路径

**Chroma 存储**:
```bash
# 默认路径
ls -la pilot/data/chromadb/

# 查看所有集合
ls -la pilot/data/chromadb/*/
```

**查看数据库向量集合**:
```bash
# 查看特定数据库的向量存储
ls -la pilot/data/chromadb/test_db_profile/
ls -la pilot/data/chromadb/test_db_profile_field/
```

#### 5.3 使用 Chroma 客户端查看

```python
from chromadb import PersistentClient

# 连接到 Chroma 数据库
client = PersistentClient(path="pilot/data/chromadb")

# 列出所有集合
collections = client.list_collections()
for collection in collections:
    print(f"Collection: {collection.name}, Count: {collection.count()}")
```

### 6. 数据迁移和备份

#### 6.1 备份向量数据

**Chroma 备份**:
```bash
# 备份整个 Chroma 数据库目录
tar -czf chroma_backup.tar.gz pilot/data/chromadb/

# 或只备份特定数据库
tar -czf test_db_vectors.tar.gz \
    pilot/data/chromadb/test_db_profile/ \
    pilot/data/chromadb/test_db_profile_field/
```

#### 6.2 迁移向量数据

**迁移到新位置**:
1. 停止服务
2. 复制 Chroma 目录到新位置
3. 更新配置文件中的 `persist_path`
4. 重启服务

```bash
# 1. 停止服务
# 2. 复制数据
cp -r pilot/data/chromadb /new/path/chromadb

# 3. 更新配置文件
# persist_path = "/new/path"

# 4. 重启服务
```

### 7. 存储空间管理

#### 7.1 查看存储大小

```bash
# 查看 Chroma 存储总大小
du -sh pilot/data/chromadb/

# 查看每个数据库的存储大小
du -sh pilot/data/chromadb/*_profile*
```

#### 7.2 清理向量数据

**删除特定数据库的向量数据**:
```python
from dbgpt_serve.datasource.service.db_summary_client import DBSummaryClient

client = DBSummaryClient(system_app=system_app)
client.delete_db_profile("test_db")  # 删除 test_db 的向量数据
```

**手动删除**:
```bash
# 删除特定数据库的向量集合
rm -rf pilot/data/chromadb/test_db_profile
rm -rf pilot/data/chromadb/test_db_profile_field
```

### 8. 配置示例

#### 8.1 使用默认路径

```toml
[rag.storage]
[rag.storage.vector]
type = "chroma"
# 不设置 persist_path，使用默认路径: {ROOT_PATH}/pilot/data/chromadb
```

#### 8.2 使用自定义路径

```toml
[rag.storage]
[rag.storage.vector]
type = "chroma"
persist_path = "/data/dbgpt/vectors"  # 绝对路径
# 最终存储路径: /data/dbgpt/vectors/chromadb
```

#### 8.3 使用环境变量

```bash
# 设置环境变量
export CHROMA_PERSIST_PATH=/data/dbgpt/vectors

# 或在 .env 文件中
echo "CHROMA_PERSIST_PATH=/data/dbgpt/vectors" >> .env
```

### 9. 注意事项

1. **路径权限**: 确保应用有读写权限
2. **磁盘空间**: 向量数据可能占用较大空间，注意监控
3. **备份策略**: 定期备份向量数据，避免数据丢失
4. **并发访问**: Chroma 支持并发访问，但建议在生产环境中使用 Milvus 等专业向量数据库
5. **性能优化**: 对于大规模数据，考虑使用 Milvus 或 Weaviate

## 总结

向量初始化是一个多步骤的过程:

1. **提取**: 从数据库连接中提取表结构信息
2. **转换**: 将表结构转换为标准化的文本格式
3. **切分**: 根据长度阈值切分过长的字段信息
4. **嵌入**: 为每个 Chunk 生成向量表示
5. **存储**: 分类存储到表向量存储和字段向量存储

**向量数据最终存储位置**:
- **Chroma (默认)**: `{persist_path}/chromadb/` 目录下
  - 表向量: `{dbname}_profile` 集合
  - 字段向量: `{dbname}_profile_field` 集合
- **Milvus**: 远程 Milvus 服务器
- **Weaviate**: 远程 Weaviate 服务器或本地持久化路径

这种设计使得:
- **高效检索**: 通过向量相似度快速找到相关表
- **灵活存储**: 大表可以分离存储，提高检索精度
- **易于维护**: 支持增量更新和版本控制
- **可扩展**: 支持多种向量存储后端

