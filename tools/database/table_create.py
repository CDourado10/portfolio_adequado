from typing import Type, List, Dict
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class ColumnDefinition(BaseModel):
    """Definition of a table column."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    constraints: List[str] = Field(default=[], description="List of constraints")

class TableCreateInput(BaseModel):
    """Input schema for TableCreate tool."""
    table_name: str = Field(..., description="Name of the table to create")
    columns: List[ColumnDefinition] = Field(..., description="List of column definitions")
    primary_key: List[str] = Field(default=[], description="List of primary key columns")
    foreign_keys: Dict[str, Dict[str, str]] = Field(
        default={},
        description="Dictionary of foreign keys {column: {ref_table: ref_column}}"
    )
    if_not_exists: bool = Field(default=True, description="Use IF NOT EXISTS")

class TableCreateTool(BaseTool):
    """Tool for creating tables in the database."""
    name: str = "CreateTable"
    description: str = (
        "Tool for creating new tables in the database. "
        "Defines columns, types, constraints, and keys."
    )
    args_schema: Type[BaseModel] = TableCreateInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        columns: List[ColumnDefinition],
        primary_key: List[str] = [],
        foreign_keys: Dict[str, Dict[str, str]] = {},
        if_not_exists: bool = True
    ) -> str:
        """Creates a new table in the database."""
        query = f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{table_name} (\n"
        
        column_defs = []
        for col in columns:
            constraints_str = ' '.join(col.constraints) if col.constraints else ''
            column_defs.append(f"    {col.name} {col.type} {constraints_str}".strip())
        
        if primary_key:
            column_defs.append(f"    PRIMARY KEY ({', '.join(primary_key)})")
        
        for col, ref in foreign_keys.items():
            for ref_table, ref_col in ref.items():
                column_defs.append(
                    f"    FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})"
                )
        
        query += ",\n".join(column_defs)
        query += "\n)"
        
        try:
            result = self.nl2sql._run(query)
            return f"Table {table_name} created successfully"
        except Exception as e:
            return f"Error creating table: {str(e)}"

if __name__ == "__main__":
    # Example usage of TableCreateTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = TableCreateTool(db_uri=db_uri)
    
    # Example 1: Create users table
    user_columns = [
        ColumnDefinition(
            name="id",
            type="SERIAL",
            constraints=["NOT NULL"]
        ),
        ColumnDefinition(
            name="username",
            type="VARCHAR(50)",
            constraints=["NOT NULL", "UNIQUE"]
        ),
        ColumnDefinition(
            name="email",
            type="VARCHAR(100)",
            constraints=["NOT NULL", "UNIQUE"]
        ),
        ColumnDefinition(
            name="active",
            type="BOOLEAN",
            constraints=["NOT NULL", "DEFAULT true"]
        ),
        ColumnDefinition(
            name="created_at",
            type="TIMESTAMP",
            constraints=["NOT NULL", "DEFAULT CURRENT_TIMESTAMP"]
        )
    ]
    
    result = tool._run(
        table_name="users",
        columns=user_columns,
        primary_key=["id"]
    )
    print("Users table creation result:")
    print(result)
    
    # Example 2: Create orders table with foreign key
    order_columns = [
        ColumnDefinition(
            name="id",
            type="SERIAL",
            constraints=["NOT NULL"]
        ),
        ColumnDefinition(
            name="user_id",
            type="INTEGER",
            constraints=["NOT NULL"]
        ),
        ColumnDefinition(
            name="total_amount",
            type="DECIMAL(10,2)",
            constraints=["NOT NULL"]
        ),
        ColumnDefinition(
            name="status",
            type="VARCHAR(20)",
            constraints=["NOT NULL", "DEFAULT 'pending'"]
        ),
        ColumnDefinition(
            name="created_at",
            type="TIMESTAMP",
            constraints=["NOT NULL", "DEFAULT CURRENT_TIMESTAMP"]
        )
    ]
    
    foreign_keys = {
        "user_id": {"users": "id"}
    }
    
    result = tool._run(
        table_name="orders",
        columns=order_columns,
        primary_key=["id"],
        foreign_keys=foreign_keys
    )
    print("\nOrders table creation result:")
    print(result)