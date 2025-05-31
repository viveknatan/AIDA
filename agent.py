from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import pandas as pd
from config import Configuration

# Try to import the database manager - use whichever is available
try:
    from database_test import DatabaseManager
    print("✅ Using database_test.py")
except ImportError:
    try:
        from database import DatabaseManager
        print("✅ Using database.py")
    except ImportError:
        raise ImportError("❌ Could not find database.py or database_test.py. Please create one of these files.")

# Import the modern LLM handler
from llm_handler import LLMHandler
from visualization import VisualizationManager

class AgentState(TypedDict):
    question: str
    schema_info: dict
    intent: dict  # Will store QuestionIntent result
    sql_query: str
    query_results: pd.DataFrame
    analysis: str
    visualization: object
    error: str

class DataAnalystAgent:
    def __init__(self, config: Configuration = None):
        self.config = config or Configuration()
        self.db_manager = DatabaseManager()
        self.llm_handler = LLMHandler(self.config)
        self.viz_manager = VisualizationManager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def get_schema(state: AgentState) -> AgentState:
            """Get database schema information"""
            try:
                state["schema_info"] = self.db_manager.get_schema_info()
            except Exception as e:
                state["error"] = f"Schema retrieval failed: {str(e)}"
            return state
        
        def classify_intent(state: AgentState) -> AgentState:
            """Classify whether the question requires database access"""
            try:
                intent_result = self.llm_handler.classify_question_intent(
                    state["question"], 
                    state["schema_info"]
                )
                
                # Convert Pydantic model to dict for state storage
                intent_dict = {
                    "is_database_related": intent_result.is_database_related,
                    "confidence": intent_result.confidence,
                    "reasoning": intent_result.reasoning,
                    "suggested_response": intent_result.suggested_response
                }
                
                state["intent"] = intent_dict
                
                # If not database-related, set a helpful response
                if not intent_result.is_database_related:
                    response = intent_result.suggested_response or (
                        "I'm an AI Data Analyst specialized in analyzing database information. "
                        f"Your question '{state['question']}' appears to be a general question that doesn't relate to the available data in our database. "
                        "I can help you analyze data from our database which contains information about "
                        f"{', '.join(state['schema_info'].keys()) if state['schema_info'] else 'various business entities'}. "
                        "Please ask questions about the data in our database, such as showing records, calculating totals, or finding patterns in the data."
                    )
                    state["analysis"] = response
                    
            except Exception as e:
                state["error"] = f"Intent classification failed: {str(e)}"
            
            return state
        
        def generate_sql(state: AgentState) -> AgentState:
            """Generate SQL query from natural language"""
            if state.get("error"):
                return state
            
            try:
                state["sql_query"] = self.llm_handler.generate_sql_query(
                    state["question"], 
                    state["schema_info"]
                )
            except Exception as e:
                state["error"] = f"SQL generation failed: {str(e)}"
            return state
        
        def execute_query(state: AgentState) -> AgentState:
            """Execute the generated SQL query"""
            if state.get("error"):
                return state
            
            try:
                state["query_results"] = self.db_manager.execute_query(state["sql_query"])
            except Exception as e:
                state["error"] = f"Query execution failed: {str(e)}"
            return state
        
        def analyze_results(state: AgentState) -> AgentState:
            """Analyze query results"""
            if state.get("error"):
                return state
            
            try:
                # Convert DataFrame to string representation for analysis
                data_str = state["query_results"].to_string()
                state["analysis"] = self.llm_handler.analyze_data(data_str, state["question"])
            except Exception as e:
                state["error"] = f"Analysis failed: {str(e)}"
            return state
        
        def create_visualization(state: AgentState) -> AgentState:
            """Create data visualization"""
            if state.get("error"):
                return state
            
            try:
                state["visualization"] = self.viz_manager.auto_visualize(
                    state["query_results"], 
                    state["question"]
                )
            except Exception as e:
                state["error"] = f"Visualization failed: {str(e)}"
            return state
        
        def should_process_database_query(state: AgentState) -> str:
            """Decide whether to proceed with database query or return early"""
            if state.get("error"):
                return "end"
            
            intent = state.get("intent", {})
            is_db_related = intent.get("is_database_related", True)
            confidence = intent.get("confidence", 0.0)
            
            # High confidence non-database questions should skip database processing
            if not is_db_related and confidence > 0.7:
                return "end"
            
            # Low confidence or database-related questions should proceed
            return "generate_sql"
        
        # Build the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("get_schema", get_schema)
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query)
        graph.add_node("analyze_results", analyze_results)
        graph.add_node("create_visualization", create_visualization)
        
        # Add edges with conditional flow
        graph.add_edge("get_schema", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            should_process_database_query,
            {
                "generate_sql": "generate_sql",
                "end": END
            }
        )
        graph.add_edge("generate_sql", "execute_query")
        graph.add_edge("execute_query", "analyze_results")
        graph.add_edge("analyze_results", "create_visualization")
        graph.add_edge("create_visualization", END)
        
        # Set entry point
        graph.set_entry_point("get_schema")
        
        return graph.compile()
    
    def process_question(self, question: str) -> AgentState:
        """Process natural language question through the agent workflow"""
        initial_state = AgentState(
            question=question,
            schema_info={},
            intent={},
            sql_query="",
            query_results=pd.DataFrame(),
            analysis="",
            visualization=None,
            error=""
        )
        
        result = self.graph.invoke(initial_state)
        return result