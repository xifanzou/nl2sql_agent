import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, inspect
from typing import Generator, List, Dict, Any, Optional

# Load environment variables
load_dotenv()

DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_NAME = os.getenv("DATABASE_NAME")

DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

class DataBase():
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.defult_table_name='kpi_benchmark'
        self.inspector = inspect(self.engine)

    def get_tables(self) -> List[str]:
        """Get all table names in the database."""
        return self.inspector.get_table_names()

    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column details for a specific table."""
        return self.inspector.get_columns(table_name)
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns for a specific table."""
        return self.inspector.get_pk_constraint(table_name)['constrained_columns']

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key constraints for a specific table."""
        return self.inspector.get_foreign_keys(table_name)
    
    def extract_schema(self) -> str:
        # get table names
        tables = self.get_tables()
        schema_text = "DATABASE SCHEMA\n==============\n\n"

        for table in tables:
            schema_text += f"Table: {table}\n"
            schema_text += "=" * (len(table) + 7) + "\n"

            # get columns & keys
            columns = self.get_table_columns(table)
            primary_keys = self.get_primary_keys(table)
            foreign_keys = {fk['constrained_columns'][0]: fk for fk in self.get_foreign_keys(table)}

            for column in columns:
                column_name = column['name']
                data_type = str(column['type'])
                nullable = "NULL" if column.get('nullable', True) else "NOT NULL"
                primary = "PRIMARY KEY" if column_name in primary_keys else ""
                
                # Check if it's a foreign key
                foreign_key_info = ""
                if column_name in foreign_keys:
                    fk = foreign_keys[column_name]
                    foreign_key_info = f"REFERENCES {fk['referred_table']}({fk['referred_columns'][0]})"

                schema_text += f"- {column_name} ({data_type}) {nullable} {primary} {foreign_key_info}\n"

            schema_text += "\n"

        return schema_text
    
    def get_schema_as_dict(self) -> Dict[str, Any]:
        """Extract schema as a structured dictionary."""
        tables = self.get_tables()
        schema_dict = {}

        for table in tables:
            columns = self.get_table_columns(table)
            primary_keys = self.get_primary_keys(table)
            foreign_keys = self.get_foreign_keys(table)
            
            schema_dict[table] = {
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys
            }

        return schema_dict

    def execute_query(self, query: str):
        """Executes a read-only SQL query on the database."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                return [dict(row) for row in result.mappings()]  # Convert result to dictionary format
        except Exception as e:
            return {"error": str(e)}