"""Automatically generate schema documentation from indexed tables."""
import pandas as pd
from typing import Dict


class SchemaBuilder:
    """Generate schema documentation dynamically."""
    
    @staticmethod
    def build_schema_prompt(dataframes: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a schema description from indexed dataframes.
        Works for ANY number of tables without manual updates.
        """
        
        if not dataframes:
            return "No tables indexed."
        
        schema = "AVAILABLE TABLES:\n\n"
        
        for table_name, df in dataframes.items():
            schema += f"Table: {table_name}\n"
            schema += f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            schema += f"  Columns: {', '.join(df.columns)}\n"
            
            # Identify key columns by name patterns
            key_cols = SchemaBuilder._find_key_columns(df)
            if key_cols:
                schema += f"  Key columns: {', '.join(key_cols)}\n"
            
            schema += "\n"
        
        return schema
    
    @staticmethod
    def _find_key_columns(df: pd.DataFrame) -> list:
        """Identify likely key columns by name patterns."""
        key_patterns = ['code', 'id', 'date', 'amount', 'euro', 'name', 'type', 'director']
        
        key_cols = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in key_patterns:
                if pattern in col_lower:
                    key_cols.append(col)
                    break
        
        return key_cols
    
    @staticmethod
    def build_grounded_system(dataframes: Dict[str, pd.DataFrame]) -> str:
        """Build complete system prompt with schema."""
        
        schema = SchemaBuilder.build_schema_prompt(dataframes)
        
        system_prompt = f"""
You are a helpful assistant answering questions about indexed data.

{schema}

CRITICAL RULES:
1. Use EXACT table and column names from above
2. Access tables: dfs['TABLE_NAME']['COLUMN_NAME']
3. NEVER guess column names
4. NEVER use dfs['COLUMN_NAME'] directly
5. For multiple tables: Use pd.merge() on common columns
6. For dates: Filter by START_DATE <= date AND (END_DATE >= date OR END_DATE is null)

Answer questions using only the tables and columns listed above.
"""
        
        return system_prompt
