"""
Modern LLM Handler using LangChain with backward compatibility
"""
from typing import Dict, Any, Optional
from config import Configuration
from models import SQLQuery, DataAnalysis, ErrorResponse

try:
    # Try LangChain imports
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
    print("✅ Using LangChain integration")
except ImportError:
    # Fallback to requests
    import requests
    import json
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available, using HTTP fallback")

class LLMHandler:
    def __init__(self, config: Optional[Configuration] = None):
        self.config = config or Configuration()
        self.config.validate()
        
        if LANGCHAIN_AVAILABLE:
            self._init_langchain()
        else:
            self._init_http_fallback()
    
    def _init_langchain(self):
        """Initialize LangChain components"""
        self.chat_llm = ChatOpenAI(
            api_key=self.config.OPENAI_API_KEY,
            model=self.config.llm_model,
            temperature=0.1,
            max_tokens=2000  # Limit response tokens
        )
        
        # Create structured LLMs for different tasks with function calling method
        self.sql_llm = self.chat_llm.with_structured_output(SQLQuery, method="function_calling")
        self.analysis_llm = self.chat_llm.with_structured_output(DataAnalysis, method="function_calling")
        
        # Define prompt templates
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL analyst. Convert natural language questions to PostgreSQL queries.

Rules:
- Generate ONLY valid PostgreSQL syntax
- Limit results to 1000 rows maximum  
- Only use SELECT statements (no INSERT, UPDATE, DELETE)
- Use table and column names exactly as shown in the schema
- Do NOT use schema prefixes in table names

Provide your confidence level based on query complexity and schema clarity."""),
            ("user", """Question: {question}

Database Schema:
{schema}

Convert this to a PostgreSQL query with explanation.""")
        ])
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst providing insights from query results.

Analyze the data and provide:
1. A concise summary of what the data shows
2. Key insights with their significance
3. Actionable recommendations
4. Notable patterns or trends

Keep insights specific and actionable."""),
            ("user", """Question: {question}

Data Results:
{data}

Analyze this data and provide comprehensive insights.""")
        ])
    
    def _init_http_fallback(self):
        """Initialize HTTP fallback"""
        self.api_key = self.config.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    
    def generate_sql_query(self, natural_language_question: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query from natural language question"""
        
        # Format schema information
        schema_description = self._format_schema_for_prompt(schema_info)
        
        try:
            if LANGCHAIN_AVAILABLE:
                return self._generate_sql_langchain(natural_language_question, schema_description)
            else:
                return self._generate_sql_http(natural_language_question, schema_description)
        
        except Exception as e:
            # Graceful fallback
            if LANGCHAIN_AVAILABLE:
                print(f"LangChain SQL generation failed: {e}")
                print("Falling back to HTTP method...")
                return self._generate_sql_http(natural_language_question, schema_description)
            else:
                raise Exception(f"SQL generation failed: {str(e)}")
    
    def _generate_sql_langchain(self, question: str, schema: str) -> str:
        """Generate SQL using LangChain structured output"""
        try:
            # Create the prompt
            formatted_prompt = self.sql_prompt.format_messages(
                question=question,
                schema=schema
            )
            
            # Get structured response
            response: SQLQuery = self.sql_llm.invoke(formatted_prompt)
            
            # Log confidence for debugging
            if self.config.DEBUG:
                print(f"SQL Generation Confidence: {response.confidence}")
                print(f"Explanation: {response.explanation}")
            
            return response.sql_query
            
        except Exception as e:
            # Create error response for debugging
            error = ErrorResponse(
                error_type="SQL_GENERATION",
                error_message=str(e),
                suggestion="Try rephrasing your question or check the database schema"
            )
            if self.config.DEBUG:
                print(f"SQL Generation Error: {error}")
            raise e
    
    def _generate_sql_http(self, question: str, schema: str) -> str:
        """HTTP fallback for SQL generation"""
        system_prompt = """You are an expert SQL analyst. Convert natural language questions to PostgreSQL queries.

Rules:
- Generate ONLY valid PostgreSQL syntax
- Limit results to 1000 rows maximum  
- Only use SELECT statements (no INSERT, UPDATE, DELETE)
- Use table and column names exactly as shown in the schema
- Do NOT use schema prefixes in table names"""

        user_prompt = f"""Question: {question}

Database Schema:
{schema}

Convert this to a PostgreSQL query."""

        response = self._make_openai_request(system_prompt, user_prompt, temperature=0.1)
        
        # Clean up response
        sql_query = response.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
    def analyze_data(self, data: str, question: str) -> str:
        """Analyze query results and provide insights"""
        
        # Truncate data if it's too long to avoid context length issues
        truncated_data = self._truncate_data_for_analysis(data)
        
        try:
            if LANGCHAIN_AVAILABLE:
                return self._analyze_data_langchain(truncated_data, question)
            else:
                return self._analyze_data_http(truncated_data, question)
        
        except Exception as e:
            # Graceful fallback
            if LANGCHAIN_AVAILABLE:
                print(f"LangChain analysis failed: {e}")
                print("Falling back to HTTP method...")
                return self._analyze_data_http(truncated_data, question)
            else:
                raise Exception(f"Analysis failed: {str(e)}")
    
    def _truncate_data_for_analysis(self, data: str) -> str:
        """Truncate data to fit within context limits"""
        max_chars = self.config.max_context_tokens * 3  # Rough estimate: 1 token ≈ 3 chars
        
        if len(data) <= max_chars:
            return data
        
        # Truncate and add note
        truncated = data[:max_chars]
        # Try to cut at a line break to avoid cutting mid-row
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.8:  # Only if we don't lose too much
            truncated = truncated[:last_newline]
        
        return truncated + f"\n\n[Note: Data truncated for analysis. Showing first {len(truncated)} characters of {len(data)} total characters.]"
    
    def _analyze_data_langchain(self, data: str, question: str) -> str:
        """Analyze data using LangChain structured output"""
        try:
            # Create the prompt
            formatted_prompt = self.analysis_prompt.format_messages(
                question=question,
                data=data
            )
            
            # Get structured response
            response: DataAnalysis = self.analysis_llm.invoke(formatted_prompt)
            
            # Format the structured response into readable text
            analysis_text = f"{response.summary}\n\n"
            
            if response.key_insights:
                analysis_text += "Key Findings:\n"
                for i, insight in enumerate(response.key_insights, 1):
                    analysis_text += f"{i}. {insight.finding}"
                    if insight.value:
                        analysis_text += f" ({insight.value})"
                    analysis_text += f" - {insight.significance}\n"
                analysis_text += "\n"
            
            if response.notable_patterns:
                analysis_text += "Notable Patterns:\n"
                for pattern in response.notable_patterns:
                    analysis_text += f"• {pattern}\n"
                analysis_text += "\n"
            
            if response.recommendations:
                analysis_text += "Recommendations:\n"
                for rec in response.recommendations:
                    analysis_text += f"• {rec}\n"
            
            return analysis_text.strip()
            
        except Exception as e:
            if self.config.DEBUG:
                print(f"Structured analysis failed: {e}")
            raise e
    
    def _analyze_data_http(self, data: str, question: str) -> str:
        """HTTP fallback for data analysis"""
        system_prompt = """You are a data analyst providing insights from query results.

Analyze the data and provide:
1. Key findings
2. Patterns or trends  
3. Notable insights
4. Summary statistics if relevant

Keep the response concise but informative."""

        user_prompt = f"""Question: {question}

Data:
{data}

Analyze this data and provide comprehensive insights."""

        return self._make_openai_request(system_prompt, user_prompt, temperature=0.3)
    
    def _make_openai_request(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Make HTTP request to OpenAI API (fallback method)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        import requests
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        
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