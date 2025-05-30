from typing import Dict, Any
from config import Config

class LLMHandler:
    def __init__(self):
        # Try langchain_openai first, then fall back to direct OpenAI
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            self.llm = ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.1
            )
            self.use_langchain = True
            print("✅ Using LangChain OpenAI")
            
        except Exception as e:
            print(f"⚠️ LangChain OpenAI failed: {e}")
            print("🔄 Falling back to direct OpenAI client...")
            
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.use_langchain = False
                print("✅ Using direct OpenAI client")
                
            except Exception as e2:
                print(f"❌ Direct OpenAI also failed: {e2}")
                raise Exception(f"Both LangChain and direct OpenAI failed. LangChain: {e}, OpenAI: {e2}")
    
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

SQL Query:
"""
        
        try:
            if self.use_langchain:
                return self._generate_sql_langchain(system_prompt, user_prompt)
            else:
                return self._generate_sql_direct(system_prompt, user_prompt)
                
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def _generate_sql_langchain(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL using LangChain"""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        sql_query = response.content.strip()
        
        # Remove potential code block formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
    def _generate_sql_direct(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL using direct OpenAI client"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Remove potential code block formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
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
            if self.use_langchain:
                return self._analyze_data_langchain(system_prompt, user_prompt)
            else:
                return self._analyze_data_direct(system_prompt, user_prompt)
                
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _analyze_data_langchain(self, system_prompt: str, user_prompt: str) -> str:
        """Analyze data using LangChain"""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        analysis_llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = analysis_llm.invoke(messages)
        return response.content.strip()
    
    def _analyze_data_direct(self, system_prompt: str, user_prompt: str) -> str:
        """Analyze data using direct OpenAI client"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()