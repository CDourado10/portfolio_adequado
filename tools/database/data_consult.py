from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool
from sqlalchemy import create_engine, MetaData, inspect

class DataConsultInput(BaseModel):
    """Input schema for DataConsult tool."""
    table_name: Optional[str] = Field(None, description="Name of the table to query. If None, returns list of available tables")
    columns: Optional[List[str]] = Field(None, description="Specific columns to query. If None, returns all columns")
    where_clause: Optional[str] = Field(None, description="WHERE condition to filter data")
    order_by: Optional[str] = Field(None, description="Columns for result ordering")
    limit: Optional[int] = Field(None, description="Maximum number of records to return")

class DataConsultTool(BaseTool):
    """Tool for querying data from the database."""
    name: str = "QueryData"
    description: str = (
        "Tool for querying data from database tables. "
        "Allows specifying columns, filter conditions, ordering, and record limits. "
        "If no table name is provided, returns a list of available tables."
    )
    args_schema: Type[BaseModel] = DataConsultInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None
    engine = None
    inspector = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)
        self.engine = create_engine(db_uri)
        self.inspector = inspect(self.engine)

    def _get_available_tables(self) -> List[Dict[str, Any]]:
        """
        Get list of available tables in the database using SQLAlchemy.
        
        Returns:
            List[Dict[str, Any]]: List of tables with their schemas
        """
        try:
            schema = self.inspector.default_schema_name
            tables = []
            for table_name in self.inspector.get_table_names(schema=schema):
                table_info = {
                    'schema': schema,
                    'table_name': table_name,
                    'type': 'table',
                    'columns': len(self.inspector.get_columns(table_name, schema=schema))
                }
                tables.append(table_info)
            return tables
        except Exception as e:
            return f"Error fetching tables: {str(e)}"

    def _get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a specific table using SQLAlchemy.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            List[Dict[str, Any]]: List of columns with their types and descriptions
        """
        try:
            columns = []
            for column in self.inspector.get_columns(table_name):
                col_info = {
                    'column_name': column['name'],
                    'data_type': str(column['type']),
                    'is_nullable': column.get('nullable', True),
                    'default': str(column.get('default', None)),
                    'primary_key': column.get('primary_key', False),
                    'foreign_keys': [],
                    'unique_constraints': []
                }
                
                # Get foreign key information
                for fk in self.inspector.get_foreign_keys(table_name):
                    if column['name'] in fk['constrained_columns']:
                        col_info['foreign_keys'].append({
                            'referred_table': fk['referred_table'],
                            'referred_columns': fk['referred_columns']
                        })
                
                # Get unique constraint information
                for const in self.inspector.get_unique_constraints(table_name):
                    if column['name'] in const['column_names']:
                        col_info['unique_constraints'].append(const['name'])
                
                columns.append(col_info)
            return columns
        except Exception as e:
            return f"Error fetching columns: {str(e)}"

    def _run(
        self, 
        table_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a SQL query based on provided parameters.
        If no table_name is provided, returns list of available tables.
        
        Args:
            table_name (str, optional): Table name. If None, returns available tables
            columns (List[str], optional): List of columns to select
            where_clause (str, optional): WHERE condition
            order_by (str, optional): ORDER BY clause
            limit (int, optional): Record limit
            
        Returns:
            List[Dict[str, Any]]: Query results or table information
        """
        # If no table name provided, return available tables
        if not table_name:
            tables = self._get_available_tables()
            if isinstance(tables, list):
                return {
                    "available_tables": tables,
                    "message": "No table specified. These are the available tables in the database."
                }
            return tables

        # If table name provided but no columns, return table structure
        if not columns and not where_clause and not order_by and not limit:
            columns = self._get_table_columns(table_name)
            if isinstance(columns, list):
                return {
                    "table_name": table_name,
                    "columns": columns,
                    "message": f"No query parameters specified. Showing structure of table '{table_name}'."
                }
            return columns

        # Build and execute regular query
        cols = "*" if not columns else ", ".join(columns)
        query = f"SELECT {cols} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
            
        if limit:
            query += f" LIMIT {limit}"
            
        try:
            result = self.nl2sql._run(query)
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

if __name__ == "__main__":
    # Example usage of DataConsultTool
    # Supports various database types:
    # SQLite: sqlite:///path/to/db.sqlite
    # PostgreSQL: postgresql://user:pass@localhost:5432/dbname
    # MySQL: mysql://user:pass@localhost:3306/dbname
    # Oracle: oracle://user:pass@localhost:1521/dbname
    # MS SQL: mssql://user:pass@localhost:1433/dbname
    
    db_uri = "sqlite:///example.db"  # Example with SQLite
    tool = DataConsultTool(db_uri=db_uri)
    
    # Example 1: Get list of available tables
    result = tool._run()
    print("Available tables in database:")
    print(result)
    
    # Example 2: Get table structure
    result = tool._run(table_name="users")
    print("\nStructure of users table:")
    print(result)
    
    # Example 3: Query all columns with limit
    result = tool._run(
        table_name="users",
        limit=5
    )
    print("\nAll columns from users table:")
    print(result)
    
    # Example 4: Query specific columns with conditions
    result = tool._run(
        table_name="orders",
        columns=["order_id", "customer_name", "total_amount"],
        where_clause="total_amount > 1000",
        order_by="total_amount DESC",
        limit=3
    )
    print("\nTop 3 orders over $1000:")
    print(result)