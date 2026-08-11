"""Helper functions for complex queries."""
import pandas as pd
from typing import Optional, Dict


class QueryHelpers:
    """Reusable functions for complex data queries."""
    
    @staticmethod
    def filter_by_date(df: pd.DataFrame, as_of_date: str, 
                       start_col: str = 'Start_date', 
                       end_col: str = 'End_date') -> pd.DataFrame:
        """Filter records valid as of a specific date."""
        if start_col not in df.columns or end_col not in df.columns:
            return df
        
        as_of = pd.Timestamp(as_of_date)
        mask = (
            (pd.to_datetime(df[start_col], errors='coerce') <= as_of) &
            (df[end_col].isna() | (pd.to_datetime(df[end_col], errors='coerce') >= as_of))
        )
        return df[mask]
    
    @staticmethod
    def convert_to_currency(df: pd.DataFrame, 
                           amount_col: str,
                           currency_col: str,
                           target_currency: str = 'EUR',
                           rates: Optional[Dict] = None) -> pd.DataFrame:
        """Convert amounts to target currency."""
        if not rates:
            rates = {
                'EUR': 1.0,
                'USD': 0.92,
                'GBP': 1.18,
                'CHF': 1.08,
                'DKK': 0.134,
                'SEK': 0.095,
                'NOK': 0.087,
            }
        
        df = df.copy()
        
        if amount_col not in df.columns or currency_col not in df.columns:
            return df
        
        def convert(row):
            try:
                amount = float(row[amount_col])
                currency = str(row[currency_col]).upper()
                rate = rates.get(currency, 1.0)
                return amount * rate
            except:
                return None
        
        df[f'{amount_col}_EUR'] = df.apply(convert, axis=1)
        return df
    
    @staticmethod
    def sum_by_type(df: pd.DataFrame,
                    group_col: str,
                    value_col: str,
                    company_code: Optional[str] = None,
                    company_col: Optional[str] = None) -> pd.DataFrame:
        """Sum amounts by type/category."""
        if company_code and company_col and company_col in df.columns:
            df = df[df[company_col].astype(str).str.strip() == company_code]
        
        if group_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        
        result = df.groupby(group_col)[value_col].sum().reset_index()
        result.columns = [group_col, f'{value_col}_TOTAL']
        return result
    
    @staticmethod
    def find_company_funds(dfs: Dict[str, pd.DataFrame],
                        company_code: str = None,
                        as_of_date: str = '2023-07-31',
                        currency: str = 'EUR',
                        fund_type: str = None) -> Optional[pd.DataFrame]:
        """
        Find all funds for a company with total by type.
        Works with ANY company code, date, and currency.
        
        Example:
        result = QueryHelpers.find_company_funds(dfs, 'EU01', '2023-07-31', 'EUR')
        result = QueryHelpers.find_company_funds(dfs, 'NT01', '2023-06-30', 'EUR')
        """
        if not company_code:
            return None
        
        # Try to find tables with funds/company data
        candidate_tables = [t for t in dfs.keys() 
                        if 'fund' in t.lower() or 'query' in t.lower() or 'company' in t.lower()]
        
        if not candidate_tables:
            return None
        
        result_list = []
        
        for table_name in candidate_tables:
            df = dfs[table_name].copy()
            
            # Find company code column (flexible matching)
            company_cols = [c for c in df.columns 
                        if 'code' in c.lower() and 'company' in c.lower()]
            if not company_cols:
                company_cols = [c for c in df.columns if 'code' in c.lower()]
            
            if not company_cols:
                continue
            
            company_col = company_cols[0]
            
            # Filter by company code (case-insensitive)
            df = df[df[company_col].astype(str).str.strip().str.upper() == company_code.upper()]
            
            if df.empty:
                continue
            
            # Filter by date
            df = QueryHelpers.filter_by_date(df, as_of_date)
            
            if df.empty:
                continue
            
            # Find amount and currency columns
            amount_cols = [c for c in df.columns if 'amount' in c.lower() or 'value' in c.lower()]
            currency_cols = [c for c in df.columns if 'currency' in c.lower()]
            
            if not amount_cols:
                continue
            
            amount_col = amount_cols[0]
            
            # Convert to target currency
            if currency_cols:
                df = QueryHelpers.convert_to_currency(df, amount_col, currency_cols[0], currency)
                amount_col = f'{amount_col}_{currency}'
            
            # Filter by fund type if specified
            if fund_type:
                type_cols = [c for c in df.columns if 'type' in c.lower()]
                if type_cols:
                    type_col = type_cols[0]
                    df = df[df[type_col].astype(str).str.upper().str.contains(fund_type.upper())]
            
            if df.empty:
                continue
            
            # Group by type
            type_cols = [c for c in df.columns if 'type' in c.lower()]
            if type_cols:
                type_col = type_cols[0]
                grouped = QueryHelpers.sum_by_type(df, type_col, amount_col)
                grouped['TABLE'] = table_name
                result_list.append(grouped)
        
        if result_list:
            return pd.concat(result_list, ignore_index=True)
        
        return None
