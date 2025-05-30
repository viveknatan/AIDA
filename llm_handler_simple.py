import requests
import json
from typing import Dict, Any
from config import Config

class LLMHandler:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        print("✅ Using HTTP-based OpenAI client")
    
    def generate_sql_query(self, natural_language_question: str, schema_info: Dict[str, Any]) -> str:
        """Convert natural language question to SQL query"""
        
        # Format schema information for the prompt
        schema_description = self._format_schema_for_prompt(schema_info)
        
        system_prompt = "You are a SQL expert that converts natural language to PostgreSQL queries."
        
        user_prompt = f"""
Convert the following natural language question into a PostgreSQL query.

Database Schema:
{schema_description}

Natural Language Question: {natural_language_question}

Instructions:
- Generate ONLY the SQL query, no explanations
- Use proper PostgreSQL syntax
- Limit results to 1000 rows maximum
- Only use SELECT statements (no INSERT, UPDATE, DELETE)
- Use table and column names exactly as shown in the schema
- Do NOT use schema prefixes in table names (e.g., use 'customers' not 'northwind.customers')
- The database connection will handle schema routing automatically

SQL Query:
"""
        
        try:
            response = self._make_openai_request(system_prompt, user_prompt, temperature=0.1)
            sql_query = response.strip()
            
            # Remove potential code block formatting
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            return sql_query.strip()
                
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def analyze_data(self, data: str, question: str) -> str:
        """Analyze query results and provide insights"""
        system_prompt = "You are a data analyst providing insights from query results."
        
        user_prompt = f"""
Analyze the following data and provide insights related to the user's question.

User Question: {question}

Data:
{data}

Provide a clear, comprehensive analysis including:
1. Key findings
2. Patterns or trends
3. Notable insights
4. Summary statistics if relevant

Keep the response concise but informative.
"""
        
        try:
            response = self._make_openai_request(system_prompt, user_prompt, temperature=0.3)
            return response.strip()
                
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _make_openai_request(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Make HTTP request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API request failed with status {response.status_code}: {response.text}")
        
        response_data = response.json()
        
        if "error" in response_data:
            raise Exception(f"OpenAI API error: {response_data['error']}")
        
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise Exception("No response choices returned from OpenAI API")
        
        return response_data["choices"][0]["message"]["content"]
    
    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for LLM prompt"""
        formatted_schema = []
        
        for table_name, table_info in schema_info.items():
            columns_str = ", ".join([
                f"{col['name']} ({col['type']}{'*' if col.get('primary_key') else ''})"
                for col in table_info["columns"]
            ])
            formatted_schema.append(f"Table: {table_name}\nColumns: {columns_str}\n")
        
        return "\n".join(formatted_schema)