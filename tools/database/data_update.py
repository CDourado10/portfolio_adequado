from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class DataUpdateInput(BaseModel):
    """Input schema for DataUpdate tool."""
    table_name: str = Field(..., description="Name of the table to update data")
    data: Dict[str, Any] = Field(..., description="Dictionary with data to be updated (column: new_value)")
    where_clause: str = Field(..., description="WHERE condition to identify records to update")
    return_updated: bool = Field(False, description="If True, returns the updated data")

class DataUpdateTool(BaseTool):
    """Tool for updating data in the database."""
    name: str = "UpdateData"
    description: str = (
        "Tool for updating data in database tables. "
        "Takes a dictionary with new values and a WHERE condition to identify records."
    )
    args_schema: Type[BaseModel] = DataUpdateInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        data: Dict[str, Any],
        where_clause: str,
        return_updated: bool = False
    ) -> str:
        """
        Updates data in a database table.
        
        Args:
            table_name (str): Table name
            data (Dict[str, Any]): Dictionary with new values
            where_clause (str): WHERE condition to identify records
            return_updated (bool): If True, returns the updated data
            
        Returns:
            str: Success or error message
        """
        # Prepare column=value pairs for query
        set_pairs = []
        for col, val in data.items():
            if isinstance(val, str):
                set_pairs.append(f"{col} = '{val}'")
            elif val is None:
                set_pairs.append(f"{col} = NULL")
            else:
                set_pairs.append(f"{col} = {val}")
        
        # Build UPDATE query
        query = f"""
            UPDATE {table_name}
            SET {', '.join(set_pairs)}
            WHERE {where_clause}
        """
        
        # Add RETURNING clause if needed
        if return_updated:
            query += " RETURNING *"
        
        # Execute query using NL2SQLTool
        try:
            result = self.nl2sql._run(query)
            if return_updated:
                return result
            return f"Data successfully updated in table {table_name}"
        except Exception as e:
            return f"Error updating data: {str(e)}"

if __name__ == "__main__":
    # Example usage of DataUpdateTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = DataUpdateTool(db_uri=db_uri)
    
    # Example 1: Update user status
    update_data = {
        "active": False,
        "last_login": "2024-02-24"
    }
    result = tool._run(
        table_name="users",
        data=update_data,
        where_clause="username = 'john_doe'",
        return_updated=True
    )
    print("Updated user data:")
    print(result)
    
    # Example 2: Update order status
    update_data = {
        "status": "completed",
        "completed_at": "2024-02-24 15:30:00"
    }
    result = tool._run(
        table_name="orders",
        data=update_data,
        where_clause="order_id = 1"
    )
    print("\nOrder update result:")
    print(result)