from typing import Type, List, Dict, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import NL2SQLTool

class IndexDefinition(BaseModel):
    """Definition of a database index."""
    name: str = Field(..., description="Index name")
    columns: List[str] = Field(..., description="List of columns to index")
    unique: bool = Field(default=False, description="Whether the index should be unique")
    method: Optional[str] = Field(default="btree", description="Index method (btree, hash, etc)")
    where: Optional[str] = Field(default=None, description="Optional WHERE condition for partial index")

class IndexManagementInput(BaseModel):
    """Input schema for IndexManagement tool."""
    table_name: str = Field(..., description="Name of the table to manage indexes")
    create_indexes: Optional[List[IndexDefinition]] = Field(
        default=None,
        description="List of indexes to create"
    )
    drop_indexes: Optional[List[str]] = Field(
        default=None,
        description="List of index names to drop"
    )
    reindex: bool = Field(
        default=False,
        description="Whether to reindex the table"
    )
    analyze: bool = Field(
        default=False,
        description="Whether to analyze table statistics after index operations"
    )

class IndexManagementTool(BaseTool):
    """Tool for managing database indexes."""
    name: str = "ManageIndexes"
    description: str = (
        "Tool for managing database indexes. "
        "Can create, drop, and reindex table indexes, and update statistics."
    )
    args_schema: Type[BaseModel] = IndexManagementInput
    db_uri: str = None
    nl2sql: NL2SQLTool = None

    def __init__(self, db_uri: str):
        super().__init__()
        self.db_uri = db_uri
        self.nl2sql = NL2SQLTool(db_uri=db_uri)

    def _run(
        self,
        table_name: str,
        create_indexes: Optional[List[IndexDefinition]] = None,
        drop_indexes: Optional[List[str]] = None,
        reindex: bool = False,
        analyze: bool = False
    ) -> str:
        """
        Manages database indexes.
        
        Args:
            table_name (str): Table to manage indexes for
            create_indexes (List[IndexDefinition], optional): Indexes to create
            drop_indexes (List[str], optional): Indexes to drop
            reindex (bool): Whether to reindex the table
            analyze (bool): Whether to analyze table statistics
            
        Returns:
            str: Success or error message
        """
        try:
            # Drop indexes if requested
            if drop_indexes:
                for index_name in drop_indexes:
                    query = f"DROP INDEX IF EXISTS {index_name}"
                    self.nl2sql._run(query)

            # Create new indexes
            if create_indexes:
                for idx in create_indexes:
                    # Build index creation query
                    unique_str = "UNIQUE" if idx.unique else ""
                    using_str = f"USING {idx.method}" if idx.method else ""
                    columns_str = ", ".join(idx.columns)
                    where_str = f"WHERE {idx.where}" if idx.where else ""
                    
                    query = f"""
                        CREATE {unique_str} INDEX {idx.name}
                        ON {table_name} {using_str} ({columns_str})
                        {where_str}
                    """.strip()
                    
                    self.nl2sql._run(query)

            # Reindex if requested
            if reindex:
                query = f"REINDEX TABLE {table_name}"
                self.nl2sql._run(query)

            # Analyze if requested
            if analyze:
                query = f"ANALYZE {table_name}"
                self.nl2sql._run(query)

            return f"Index operations completed successfully on table {table_name}"
        except Exception as e:
            return f"Error managing indexes: {str(e)}"

if __name__ == "__main__":
    # Example usage of IndexManagementTool
    db_uri = "postgresql://user:pass@localhost:5432/dbname"
    tool = IndexManagementTool(db_uri=db_uri)
    
    # Example 1: Create indexes on users table
    new_indexes = [
        IndexDefinition(
            name="idx_users_email",
            columns=["email"],
            unique=True
        ),
        IndexDefinition(
            name="idx_users_status_created",
            columns=["status", "created_at"],
            method="btree",
            where="status = 'active'"
        )
    ]
    result = tool._run(
        table_name="users",
        create_indexes=new_indexes
    )
    print("Create indexes result:")
    print(result)
    
    # Example 2: Drop and reindex
    result = tool._run(
        table_name="users",
        drop_indexes=["old_index_name"],
        reindex=True,
        analyze=True
    )
    print("\nDrop and reindex result:")
    print(result)
