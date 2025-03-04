from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class DataDeleteInput(BaseModel):
    """Input schema for DataDelete tool."""
    table_name: str = Field(..., description="Name of the table to delete data from")
    where_clause: str = Field(..., description="WHERE condition to identify records to delete")
    return_deleted: bool = Field(False, description="If True, returns the deleted data (using RETURNING)")
    cascade: bool = Field(False, description="If True, deletes related records in other tables (CASCADE)")

class DataDeleteTool(BaseTool):
    """Tool for deleting data from the database."""
    name: str = "DeleteData"
    description: str = (
        "Tool for deleting data from database tables. "
        "Requires a WHERE condition to identify records to delete. "
        "Optionally can return deleted data and use CASCADE."
    )
    args_schema: Type[BaseModel] = DataDeleteInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        where_clause: str,
        return_deleted: bool = False,
        cascade: bool = False
    ) -> str:
        """
        Deletes data from a database table.
        
        Args:
            table_name (str): Table name
            where_clause (str): WHERE condition to identify records
            return_deleted (bool): If True, returns the deleted data
            cascade (bool): If True, uses CASCADE to delete related records
            
        Returns:
            str: Success or error message
        """
        # Build DELETE query
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        
        # Add CASCADE if needed
        if cascade:
            query += " CASCADE"
            
        # Add RETURNING clause if needed
        if return_deleted:
            query += " RETURNING *"
        
        # Execute query using NL2SQLTool
        try:
            result = self.nl2sql._run(query)
            if return_deleted:
                return result
            return f"Data successfully deleted from table {table_name}"
        except Exception as e:
            return f"Error deleting data: {str(e)}"

if __name__ == "__main__":
    # Example usage of DataDeleteTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = DataDeleteTool(db_uri=db_uri)
    
    # Example 1: Delete inactive users
    result = tool._run(
        table_name="users",
        where_clause="active = false AND last_login < '2023-01-01'",
        return_deleted=True
    )
    print("Deleted inactive users:")
    print(result)
    
    # Example 2: Delete order with cascade
    result = tool._run(
        table_name="orders",
        where_clause="order_id = 1",
        cascade=True
    )
    print("\nOrder deletion result:")
    print(result)