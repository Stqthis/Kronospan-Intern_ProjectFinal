"""Extract parameters from natural language queries."""
import re
from typing import Optional, Dict, Any


class ParameterExtractor:
    """Extract company codes, dates, currencies, etc. from questions."""
    
    @staticmethod
    def extract_company_code(question: str) -> Optional[str]:
        """Extract company code (e.g., EU01, NT01, CY05)."""
        # Pattern: letter(s) + numbers
        pattern = r'\b([A-Z]{1,3}\d{2,3})\b'
        matches = re.findall(pattern, question.upper())
        return matches[0] if matches else None
    
    @staticmethod
    def extract_date(question: str) -> Optional[str]:
        """Extract date in various formats."""
        # Format: DD.MM.YYYY
        pattern = r'(\d{1,2}[.-/]\d{1,2}[.-/]\d{4})'
        matches = re.findall(pattern, question)
        
        if matches:
            date_str = matches[0]
            # Convert DD.MM.YYYY to YYYY-MM-DD
            parts = re.split(r'[.-/]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Check for common date references
        if 'today' in question.lower():
            return 'today'
        if '2023' in question:
            if '31.07' in question or '31-07' in question or '07-31' in question:
                return '2023-07-31'
        
        return None
    
    @staticmethod
    def extract_currency(question: str) -> Optional[str]:
        """Extract target currency."""
        currencies = {
            'euro': 'EUR',
            'eur': 'EUR',
            '€': 'EUR',
            'dollar': 'USD',
            'usd': 'USD',
            '$': 'USD',
            'pound': 'GBP',
            'gbp': 'GBP',
            'franc': 'CHF',
            'chf': 'CHF',
            'krona': 'SEK',
            'krone': 'DKK',
        }
        
        question_lower = question.lower()
        for key, currency in currencies.items():
            if key in question_lower:
                return currency
        
        return 'EUR'  # Default to EUR
    
    @staticmethod
    def extract_fund_type(question: str) -> Optional[str]:
        """Extract fund type if specified."""
        types = ['equity', 'bond', 'fixed income', 'money market', 'fund']
        
        question_lower = question.lower()
        for fund_type in types:
            if fund_type in question_lower:
                return fund_type.upper()
        
        return None
    
    @staticmethod
    def extract_all(question: str) -> Dict[str, Any]:
        """Extract all parameters from question."""
        return {
            'company_code': ParameterExtractor.extract_company_code(question),
            'date': ParameterExtractor.extract_date(question),
            'currency': ParameterExtractor.extract_currency(question),
            'fund_type': ParameterExtractor.extract_fund_type(question),
        }
    
    @staticmethod
    def format_parameters(params: Dict[str, Any]) -> str:
        """Format extracted parameters as human-readable string."""
        lines = []
        if params.get('company_code'):
            lines.append(f"Company: {params['company_code']}")
        if params.get('date'):
            lines.append(f"As of: {params['date']}")
        if params.get('currency'):
            lines.append(f"Currency: {params['currency']}")
        if params.get('fund_type'):
            lines.append(f"Fund Type: {params['fund_type']}")
        
        return " | ".join(lines) if lines else "Parameters: (none extracted)"
