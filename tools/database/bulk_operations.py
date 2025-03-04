from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class BulkOperationsInput(BaseModel):
    """Input schema for BulkOperations tool."""
    table_name: str = Field(..., description="Name of the table for bulk operations")
    operation: str = Field(
        ...,
        description="Operation type: 'insert', 'update', or 'delete'"
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description="List of records for insert/update operations"
    )
    where_clause: str = Field(
        default=None,
        description="WHERE condition for update/delete operations"
    )
    batch_size: int = Field(
        default=1000,
        description="Number of records to process in each batch"
    )
    return_count: bool = Field(
        default=True,
        description="Whether to return the number of affected records"
    )

class BulkOperationsTool(BaseTool):
    """Tool for performing bulk operations in the database."""
    name: str = "BulkOperations"
    description: str = (
        "Tool for performing bulk operations (insert/update/delete) in database tables. "
        "Handles large datasets efficiently using batching."
    )
    args_schema: Type[BaseModel] = BulkOperationsInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        operation: str,
        data: List[Dict[str, Any]],
        where_clause: str = None,
        batch_size: int = 1000,
        return_count: bool = True
    ) -> str:
        """
        Performs bulk operations on a database table.
        
        Args:
            table_name (str): Target table
            operation (str): Operation type (insert/update/delete)
            data (List[Dict]): Records to process
            where_clause (str, optional): WHERE condition for update/delete
            batch_size (int): Batch size for processing
            return_count (bool): Whether to return affected record count
            
        Returns:
            str: Success or error message
        """
        try:
            total_affected = 0
            
            if operation.lower() == 'insert':
                total_affected = self._bulk_insert(table_name, data, batch_size)
            elif operation.lower() == 'update':
                total_affected = self._bulk_update(table_name, data, where_clause, batch_size)
            elif operation.lower() == 'delete':
                total_affected = self._bulk_delete(table_name, where_clause)
            else:
                return f"Invalid operation: {operation}"
            
            if return_count:
                return f"Bulk {operation} completed. {total_affected} records affected."
            return f"Bulk {operation} completed successfully"
            
        except Exception as e:
            return f"Error in bulk operation: {str(e)}"

    def _bulk_insert(self, table_name: str, data: List[Dict[str, Any]], batch_size: int) -> int:
        """Performs bulk insert operation."""
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if not batch:
                continue
                
            # Prepare columns and values
            columns = list(batch[0].keys())
            values_list = []
            
            for record in batch:
                values = []
                for col in columns:
                    val = record.get(col)
                    if isinstance(val, str):
                        values.append(f"'{val}'")
                    elif val is None:
                        values.append('NULL')
                    else:
                        values.append(str(val))
                values_list.append(f"({', '.join(values)})")
            
            query = f"""
                INSERT INTO {table_name} 
                ({', '.join(columns)})
                VALUES {', '.join(values_list)}
            """
            
            self.nl2sql._run(query)
            total_inserted += len(batch)
            
        return total_inserted

    def _bulk_update(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        where_clause: str,
        batch_size: int
    ) -> int:
        """Performs bulk update operation."""
        total_updated = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if not batch:
                continue
            
            # Prepare SET clause
            set_items = []
            for col, val in batch[0].items():
                if isinstance(val, str):
                    set_items.append(f"{col} = '{val}'")
                elif val is None:
                    set_items.append(f"{col} = NULL")
                else:
                    set_items.append(f"{col} = {val}")
            
            query = f"""
                UPDATE {table_name}
                SET {', '.join(set_items)}
                WHERE {where_clause}
            """
            
            result = self.nl2sql._run(query)
            if isinstance(result, list):
                total_updated += len(result)
            
        return total_updated

    def _bulk_delete(self, table_name: str, where_clause: str) -> int:
        """Performs bulk delete operation."""
        query = f"""
            DELETE FROM {table_name}
            WHERE {where_clause}
            RETURNING 1
        """
        
        result = self.nl2sql._run(query)
        return len(result) if isinstance(result, list) else 0

if __name__ == "__main__":
    # Example usage of BulkOperationsTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = BulkOperationsTool(db_uri=db_uri)
    
    # Example 1: Bulk insert users
    users_data = [
        {"username": f"user{i}", "email": f"user{i}@example.com"}
        for i in range(1, 1001)
    ]
    result = tool._run(
        table_name="users",
        operation="insert",
        data=users_data,
        batch_size=100
    )
    print("Bulk insert result:")
    print(result)
    
    # Example 2: Bulk update status
    update_data = [{"status": "inactive", "last_updated": "2024-02-24"}]
    result = tool._run(
        table_name="users",
        operation="update",
        data=update_data,
        where_clause="last_login < '2023-12-31'",
        return_count=True
    )
    print("\nBulk update result:")
    print(result)
    
    # Example 3: Bulk delete old records
    result = tool._run(
        table_name="logs",
        operation="delete",
        data=[],  # Not needed for delete
        where_clause="created_at < '2023-01-01'",
        return_count=True
    )
    print("\nBulk delete result:")
    print(result)
