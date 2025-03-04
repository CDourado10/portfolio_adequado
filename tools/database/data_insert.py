from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class DataInsertInput(BaseModel):
    """Input schema for DataInsert tool."""
    table_name: str = Field(..., description="Name of the table to insert data")
    data: Dict[str, Any] = Field(..., description="Dictionary with data to be inserted (column: value)")
    return_inserted: bool = Field(False, description="If True, returns the inserted data")

class DataInsertTool(BaseTool):
    """Tool for inserting data into the database."""
    name: str = "InsertData"
    description: str = (
        "Tool for inserting data into database tables. "
        "Takes a dictionary with data to be inserted and optionally returns the inserted data."
    )
    args_schema: Type[BaseModel] = DataInsertInput
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
        return_inserted: bool = False
    ) -> str:
        """
        Inserts data into a database table.
        
        Args:
            table_name (str): Table name
            data (Dict[str, Any]): Dictionary with data to be inserted
            return_inserted (bool): If True, returns the inserted data
            
        Returns:
            str: Success or error message
        """
        # Prepare columns and values for query
        columns = list(data.keys())
        values = list(data.values())
        
        # Format values for SQL query
        formatted_values = []
        for val in values:
            if isinstance(val, str):
                formatted_values.append(f"'{val}'")
            elif val is None:
                formatted_values.append('NULL')
            else:
                formatted_values.append(str(val))
        
        # Build INSERT query
        query = f"""
            INSERT INTO {table_name} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(formatted_values)})
        """
        
        # Add RETURNING clause if needed
        if return_inserted:
            query += " RETURNING *"
        
        # Execute query using NL2SQLTool
        try:
            result = self.nl2sql._run(query)
            if return_inserted:
                return result
            return f"Data successfully inserted into table {table_name}"
        except Exception as e:
            return f"Error inserting data: {str(e)}"

if __name__ == "__main__":
    # Example usage of DataInsertTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = DataInsertTool(db_uri=db_uri)
    
    # Example 1: Insert a new user
    user_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "created_at": "2024-02-24",
        "active": True
    }
    result = tool._run(
        table_name="users",
        data=user_data,
        return_inserted=True
    )
    print("Inserted user data:")
    print(result)
    
    # Example 2: Insert an order
    order_data = {
        "customer_id": 1,
        "total_amount": 1500.50,
        "order_date": "2024-02-24",
        "status": "pending"
    }
    result = tool._run(
        table_name="orders",
        data=order_data
    )
    print("\nOrder insertion result:")
    print(result)