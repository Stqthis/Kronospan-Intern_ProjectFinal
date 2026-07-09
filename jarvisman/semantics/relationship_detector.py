"""Detect relationships between tables for smart joins."""
import pandas as pd
from typing import Dict, List, Tuple, Set


class RelationshipDetector:
    """Automatically detect relationships between tables."""
    
    @staticmethod
    def find_relationships(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """
        Find potential relationships between tables.
        
        Returns:
            {
                'table1:table2': [
                    {
                        'left_table': 'table1',
                        'right_table': 'table2',
                        'left_column': 'COMPANY_CODE',
                        'right_column': 'CODE',
                        'strength': 0.95,  # How confident we are
                        'type': 'foreign_key'  # or 'semantic_match'
                    }
                ]
            }
        """
        relationships = {}
        table_names = list(dataframes.keys())
        
        # Compare each pair of tables
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                rels = RelationshipDetector._detect_pair(
                    table1, dataframes[table1],
                    table2, dataframes[table2]
                )
                
                if rels:
                    key = f"{table1}→{table2}"
                    relationships[key] = rels
        
        return relationships
    
    @staticmethod
    def _detect_pair(table1_name: str, df1: pd.DataFrame,
                     table2_name: str, df2: pd.DataFrame) -> List[Dict]:
        """Detect relationships between two specific tables."""
        relationships = []
        
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        # Strategy 1: Exact column name matches
        for col in cols1 & cols2:
            if RelationshipDetector._is_key_column(col):
                # Check if values match
                match_strength = RelationshipDetector._calculate_match_strength(
                    df1[col], df2[col]
                )
                
                if match_strength > 0.7:
                    relationships.append({
                        'left_table': table1_name,
                        'right_table': table2_name,
                        'left_column': col,
                        'right_column': col,
                        'strength': match_strength,
                        'type': 'exact_match'
                    })
        
        # Strategy 2: Similar column names (e.g., CODE vs COMPANY_CODE)
        for col1 in cols1:
            for col2 in cols2:
                if RelationshipDetector._columns_similar(col1, col2):
                    match_strength = RelationshipDetector._calculate_match_strength(
                        df1[col1], df2[col2]
                    )
                    
                    if match_strength > 0.8:
                        relationships.append({
                            'left_table': table1_name,
                            'right_table': table2_name,
                            'left_column': col1,
                            'right_column': col2,
                            'strength': match_strength,
                            'type': 'semantic_match'
                        })
        
        # Strategy 3: Common ID patterns
        for col1 in cols1:
            for col2 in cols2:
                if RelationshipDetector._is_id_match(col1, col2):
                    match_strength = RelationshipDetector._calculate_match_strength(
                        df1[col1], df2[col2]
                    )
                    
                    if match_strength > 0.75:
                        relationships.append({
                            'left_table': table1_name,
                            'right_table': table2_name,
                            'left_column': col1,
                            'right_column': col2,
                            'strength': match_strength,
                            'type': 'id_pattern_match'
                        })
        
        # Remove duplicates, keep strongest
        return RelationshipDetector._deduplicate_relationships(relationships)
    
    @staticmethod
    def _is_key_column(col_name: str) -> bool:
        """Check if column looks like a key/ID column."""
        key_patterns = ['ID', 'CODE', 'KEY', '_ID', '_CODE']
        col_upper = col_name.upper()
        return any(pattern in col_upper for pattern in key_patterns)
     
    @staticmethod
    def _columns_similar(col1: str, col2: str) -> bool:
        """Check if two column names are semantically similar."""
        # Remove common prefixes/suffixes
        def normalize(s):
            s = s.upper()
            for prefix in ['TABLE_', 'COMPANY_', 'FK_', 'PK_']:
                s = s.replace(prefix, '')
            return s
        
        norm1 = normalize(col1)
        norm2 = normalize(col2)
        
        # Check if one contains the other
        return norm1 in norm2 or norm2 in norm1 or norm1 == norm2
    
    @staticmethod
    def _is_id_match(col1: str, col2: str) -> bool:
        """Check for ID column patterns."""
        col1_parts = col1.upper().split('_')
        col2_parts = col2.upper().split('_')
        
        # If both end with ID/CODE, they might match
        if col1_parts[-1] in ['ID', 'CODE'] and col2_parts[-1] in ['ID', 'CODE']:
            # Check if other parts match
            return col1_parts[:-1] == col2_parts[:-1] or \
                   col1_parts[:-1][-1] == col2_parts[:-1][-1]
        
        return False
    
    @staticmethod
    def _calculate_match_strength(s1: pd.Series, s2: pd.Series) -> float:
        """Calculate how well two columns match (0-1)."""
        try:
            # Convert to string for comparison
            v1 = set(s1.dropna().astype(str).unique())
            v2 = set(s2.dropna().astype(str).unique())
            
            if not v1 or not v2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(v1 & v2)
            union = len(v1 | v2)
            
            if union == 0:
                return 0.0
            
            return intersection / union
        except:
            return 0.0
    
    @staticmethod
    def _deduplicate_relationships(rels: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships, keep strongest."""
        seen = {}
        
        for rel in sorted(rels, key=lambda x: x['strength'], reverse=True):
            key = (rel['left_column'], rel['right_column'])
            
            if key not in seen:
                seen[key] = rel
        
        return list(seen.values())
    
    @staticmethod
    def create_join_query(df1: pd.DataFrame, df2: pd.DataFrame,
                         left_on: str, right_on: str) -> pd.DataFrame:
        """Create a joined dataframe based on detected relationship."""
        try:
            return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how='inner')
        except Exception as e:
            print(f"Error joining tables: {e}")
            return None
