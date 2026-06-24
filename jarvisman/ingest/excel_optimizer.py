"""Fast Excel indexing with caching and optimization."""
import os
import pickle
import hashlib
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Optional


class FastExcelIndexer:
    """Optimized Excel indexing with caching."""
    
    def __init__(self, cache_dir='./index_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Get MD5 hash of file to detect changes."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error computing hash: {e}")
            return None
    
    def is_cached(self, file_path: str) -> bool:
        """Check if file is already indexed."""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        return os.path.exists(cache_file)
    
    def load_from_cache(self, file_path: str) -> Optional[Tuple]:
        """Load cached index."""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def save_to_cache(self, file_path: str, data: Tuple) -> bool:
        """Save index to cache."""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    @staticmethod
    def load_excel_smart(file_path: str, max_rows: int = 10000, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load Excel file efficiently."""
        print(f"Loading {file_path}...")
        
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=0,
                nrows=max_rows,
                usecols=columns
            )
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        print("Cleaning data...")
        
        df = df.drop_duplicates()
        df = df.dropna(thresh=len(df.columns) * 0.5)
        
        print(f"✓ After cleaning: {len(df)} rows")
        return df
    
    @staticmethod
    def create_chunks(df: pd.DataFrame, min_length: int = 10) -> List[str]:
        """Create text chunks from dataframe."""
        print("Creating chunks...")
        
        chunks = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            chunk_parts = []
            
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    chunk_parts.append(f"{col}: {str(value)}")
            
            chunk = " | ".join(chunk_parts)
            
            if len(chunk) > min_length:
                chunks.append(chunk)
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def index_excel(self, file_path: str, max_rows: int = 10000,
                   columns: Optional[List[str]] = None) -> Tuple:
        """Index Excel file with optimization."""
        print(f"\nIndexing: {os.path.basename(file_path)}")
        
        if self.is_cached(file_path):
            print("📦 Loading from cache (fast!)...")
            result = self.load_from_cache(file_path)
            if result:
                print("✓ Cache loaded!\n")
                return result
        
        df = self.load_excel_smart(file_path, max_rows, columns)
        if df is None or df.empty:
            return None
        
        df = self.clean_data(df)
        chunks = self.create_chunks(df)
        
        if not chunks:
            print("No chunks created!")
            return None
        
        metadata = {
            'file': os.path.basename(file_path),
            'rows': len(df),
            'chunks': len(chunks),
            'columns': list(df.columns)
        }
        
        print("💾 Saving to cache...")
        self.save_to_cache(file_path, (chunks, metadata))
        print("✓ Complete!\n")
        
        return chunks, metadata
