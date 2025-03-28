
# nl2sql_agent

[中文版本](README.md)

## Overview

`NL2SQL_AGENT` is a lightweight natural language to SQL (NL2SQL) agent designed to help users query PostgreSQL databases using natural language. This agent combines document processing, vector retrieval, LLM, and database interaction to provide users with an efficient and convenient database query experience.

This tool is based on SiliconFlow-driven LLM, and sample code for TensorFlow-driven LLM is provided in the core code files.

This tool is suitable for users who frequently need to query databases but are not familiar with SQL syntax.

## Core Features

1.  **Natural Language to SQL Conversion**: Users can query the database using natural language descriptions, and the tool will automatically generate the corresponding SQL query statements.

2.  **Multi-Document Format Support**: Supports uploading, storing, and parsing documents in various formats (such as CSV, TXT, Markdown).

3.  **Interactive Query**: Provides an interactive command-line interface where users can input queries and obtain results in real-time.

4.  **Efficient Vector Retrieval**: Utilizes the Faiss library to achieve efficient semantic search and vector recall.


## Demo

![Video Demo](assets/demo.mov)

## Installation and Usage

### Install Dependencies

Ensure that all necessary Python packages are installed. You can use the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

### Environment Configuration
1. Create a .env file.

2. Add the database-related parameters to it, such as:
```
# Database
DATABASE_NAME="postgres"
DATABASE_USER="postgres"
DATABASE_PASSWORD="postgrespwd"
DATABASE_HOST="localhost"
DATABASE_PORT=5432
```
### Run the Program

```bash
treamlit run app/web.py
```