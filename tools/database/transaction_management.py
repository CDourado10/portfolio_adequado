from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class TransactionOperation(BaseModel):
    """Definition of a transaction operation."""
    query: str = Field(..., description="SQL query to execute")
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters"
    )

class TransactionManagementInput(BaseModel):
    """Input schema for TransactionManagement tool."""
    operations: List[TransactionOperation] = Field(
        ...,
        description="List of operations to execute in transaction"
    )
    savepoints: Optional[List[str]] = Field(
        default=None,
        description="List of savepoint names to create during transaction"
    )
    rollback_to: Optional[str] = Field(
        default=None,
        description="Savepoint name to rollback to if error occurs"
    )

class TransactionManagementTool(BaseTool):
    """Tool for managing database transactions."""
    name: str = "ManageTransactions"
    description: str = (
        "Tool for managing database transactions. "
        "Supports multiple operations, savepoints, and partial rollbacks."
    )
    args_schema: Type[BaseModel] = TransactionManagementInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        operations: List[TransactionOperation],
        savepoints: Optional[List[str]] = None,
        rollback_to: Optional[str] = None
    ) -> str:
        """
        Executes a series of operations in a transaction.
        
        Args:
            operations (List[TransactionOperation]): Operations to execute
            savepoints (List[str], optional): Savepoints to create
            rollback_to (str, optional): Savepoint to rollback to on error
            
        Returns:
            str: Success or error message
        """
        try:
            # Start transaction
            self.nl2sql._run("BEGIN")
            
            # Create initial savepoint if needed
            if savepoints:
                self.nl2sql._run(f"SAVEPOINT {savepoints[0]}")
                current_savepoint = 0
            
            # Execute operations
            results = []
            for i, op in enumerate(operations):
                try:
                    # Create next savepoint if available
                    if savepoints and i > 0 and current_savepoint < len(savepoints) - 1:
                        current_savepoint += 1
                        self.nl2sql._run(f"SAVEPOINT {savepoints[current_savepoint]}")
                    
                    # Execute operation
                    result = self.nl2sql._run(op.query)
                    results.append(result)
                    
                except Exception as e:
                    # Handle operation error
                    if rollback_to and rollback_to in savepoints:
                        self.nl2sql._run(f"ROLLBACK TO SAVEPOINT {rollback_to}")
                        return f"Error in operation {i+1}, rolled back to savepoint {rollback_to}: {str(e)}"
                    else:
                        self.nl2sql._run("ROLLBACK")
                        return f"Error in operation {i+1}, transaction rolled back: {str(e)}"
            
            # Commit transaction
            self.nl2sql._run("COMMIT")
            return f"Transaction completed successfully. {len(operations)} operations executed."
            
        except Exception as e:
            # Handle transaction error
            self.nl2sql._run("ROLLBACK")
            return f"Transaction error: {str(e)}"

if __name__ == "__main__":
    # Example usage of TransactionManagementTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = TransactionManagementTool(db_uri=db_uri)
    
    # Example 1: Transfer money between accounts
    transfer_ops = [
        TransactionOperation(
            query="UPDATE accounts SET balance = balance - 100 WHERE id = 1"
        ),
        TransactionOperation(
            query="UPDATE accounts SET balance = balance + 100 WHERE id = 2"
        )
    ]
    result = tool._run(
        operations=transfer_ops,
        savepoints=["before_transfer"]
    )
    print("Money transfer result:")
    print(result)
    
    # Example 2: Complex order processing with savepoints
    order_ops = [
        TransactionOperation(
            query="INSERT INTO orders (customer_id, total) VALUES (1, 500) RETURNING id"
        ),
        TransactionOperation(
            query="INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 1, 2)"
        ),
        TransactionOperation(
            query="UPDATE inventory SET stock = stock - 2 WHERE product_id = 1"
        )
    ]
    result = tool._run(
        operations=order_ops,
        savepoints=["after_order", "after_items"],
        rollback_to="after_order"
    )
    print("\nOrder processing result:")
    print(result)
    
    # Example 3: Batch user status update
    update_ops = [
        TransactionOperation(
            query="""
                UPDATE users 
                SET status = 'inactive', 
                    last_updated = CURRENT_TIMESTAMP 
                WHERE last_login < '2023-12-31'
            """
        ),
        TransactionOperation(
            query="INSERT INTO audit_log (action, details) VALUES ('bulk_update', 'Deactivated inactive users')"
        )
    ]
    result = tool._run(operations=update_ops)
    print("\nBatch update result:")
    print(result)
