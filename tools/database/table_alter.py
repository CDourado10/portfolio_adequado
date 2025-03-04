from typing import Type, List, Dict, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class ColumnDefinition(BaseModel):
    """Definition of a table column."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    constraints: List[str] = Field(default=[], description="List of constraints")

class TableAlterInput(BaseModel):
    """Input schema for TableAlter tool."""
    table_name: str = Field(..., description="Name of the table to alter")
    add_columns: Optional[List[ColumnDefinition]] = Field(
        default=None,
        description="List of columns to add"
    )
    drop_columns: Optional[List[str]] = Field(
        default=None,
        description="List of column names to drop"
    )
    rename_columns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of old_name: new_name for columns to rename"
    )
    add_constraints: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of constraint_name: definition to add"
    )
    drop_constraints: Optional[List[str]] = Field(
        default=None,
        description="List of constraint names to drop"
    )
    rename_table: Optional[str] = Field(
        default=None,
        description="New name for the table"
    )

class TableAlterTool(BaseTool):
    """Tool for altering table structure in the database."""
    name: str = "AlterTable"
    description: str = (
        "Tool for modifying database table structure. "
        "Can add/drop columns, rename columns/tables, and manage constraints."
    )
    args_schema: Type[BaseModel] = TableAlterInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        add_columns: Optional[List[ColumnDefinition]] = None,
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        add_constraints: Optional[Dict[str, str]] = None,
        drop_constraints: Optional[List[str]] = None,
        rename_table: Optional[str] = None
    ) -> str:
        """
        Alters a database table structure.
        
        Args:
            table_name (str): Table to alter
            add_columns (List[ColumnDefinition], optional): Columns to add
            drop_columns (List[str], optional): Columns to drop
            rename_columns (Dict[str, str], optional): Columns to rename
            add_constraints (Dict[str, str], optional): Constraints to add
            drop_constraints (List[str], optional): Constraints to drop
            rename_table (str, optional): New table name
            
        Returns:
            str: Success or error message
        """
        try:
            # Rename table if requested
            if rename_table:
                query = f"ALTER TABLE {table_name} RENAME TO {rename_table}"
                self.nl2sql._run(query)
                table_name = rename_table

            # Add new columns
            if add_columns:
                for col in add_columns:
                    constraints_str = ' '.join(col.constraints) if col.constraints else ''
                    query = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col.type} {constraints_str}".strip()
                    self.nl2sql._run(query)

            # Drop columns
            if drop_columns:
                for col in drop_columns:
                    query = f"ALTER TABLE {table_name} DROP COLUMN {col}"
                    self.nl2sql._run(query)

            # Rename columns
            if rename_columns:
                for old_name, new_name in rename_columns.items():
                    query = f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}"
                    self.nl2sql._run(query)

            # Add constraints
            if add_constraints:
                for name, definition in add_constraints.items():
                    query = f"ALTER TABLE {table_name} ADD CONSTRAINT {name} {definition}"
                    self.nl2sql._run(query)

            # Drop constraints
            if drop_constraints:
                for constraint in drop_constraints:
                    query = f"ALTER TABLE {table_name} DROP CONSTRAINT {constraint}"
                    self.nl2sql._run(query)

            return f"Table {table_name} altered successfully"
        except Exception as e:
            return f"Error altering table: {str(e)}"

if __name__ == "__main__":
    # Example usage of TableAlterTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = TableAlterTool(db_uri=db_uri)
    
    # Example 1: Add new columns to users table
    new_columns = [
        ColumnDefinition(
            name="last_login",
            type="TIMESTAMP",
            constraints=["DEFAULT CURRENT_TIMESTAMP"]
        ),
        ColumnDefinition(
            name="status",
            type="VARCHAR(20)",
            constraints=["DEFAULT 'active'", "CHECK (status IN ('active', 'inactive'))"]
        )
    ]
    result = tool._run(
        table_name="users",
        add_columns=new_columns
    )
    print("Add columns result:")
    print(result)
    
    # Example 2: Add and modify constraints
    result = tool._run(
        table_name="users",
        add_constraints={
            "uk_users_email": "UNIQUE (email)",
            "chk_users_age": "CHECK (age >= 18)"
        },
        drop_constraints=["old_constraint"]
    )
    print("\nConstraints modification result:")
    print(result)
    
    # Example 3: Rename columns and table
    result = tool._run(
        table_name="users",
        rename_columns={"username": "login", "fullname": "name"},
        rename_table="system_users"
    )
    print("\nRename operation result:")
    print(result)
