# nl2sql_agent

[English Version](README_en.md)

## 概述

`NL2SQL_AGENT`是一个轻量级的自然语言到 SQL (NL2SQL) 智能体，旨在帮助用户通过自然语言查询 PostgreSQL 数据库。该智能体结合了文档处理、向量检索、LLM 和数据库交互，为用户提供高效、便捷的数据库查询体验。

本工具基于SiliconFLow驱动LLM，核心代码文件中提供了TensorFlow驱动LLM的样本代码。

工具适用于需要频繁进行数据库查询但不熟悉SQL语法的用户。



## 核心功能

1. **自然语言到 SQL 的转换**：用户可以通过自然语言描述来查询数据库，工具会自动生成相应的 SQL 查询语句。


2. **多文档格式支持**：支持多种格式的文档（如 CSV、TXT、Markdown）上传、存储与解析。

3. **交互式查询**：提供了一个交互式的命令行界面，用户可以实时输入查询并获取结果。

4. **高效向量检索**：利用 Faiss 库实现高效的语义搜索和向量召回功能。


## 安装与使用

### 安装依赖

确保安装了所有必要的 Python 包。可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```
### 环境配置

1. 创建`.env`文件

2. 在其中添加数据库相关参数，如：
```
# Database
DATABASE_NAME="postgres"
DATABASE_USER="postgres"
DATABASE_PASSWORD="postgrespwd"
DATABASE_HOST="localhost"
DATABASE_PORT=5432
```

### 运行程序

```bash
streamlit run app/web.py
```