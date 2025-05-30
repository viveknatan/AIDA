#!/usr/bin/env python3

"""
Test script to verify all imports work correctly
Run this before running the main app
"""

print("🧪 Testing imports...")

try:
    print("1. Testing config...")
    from config import Config
    print("   ✅ Config imported successfully")
    
    print("2. Testing database_test...")
    from database_test import DatabaseManager
    print("   ✅ DatabaseManager imported successfully")
    
    print("3. Testing llm_handler_simple...")
    from llm_handler_simple import LLMHandler
    print("   ✅ LLMHandler imported successfully")
    
    print("4. Testing visualization...")
    from visualization import VisualizationManager
    print("   ✅ VisualizationManager imported successfully")
    
    print("5. Testing langgraph...")
    from langgraph.graph import StateGraph, END
    print("   ✅ LangGraph imported successfully")
    
    print("6. Testing agent...")
    from agent import DataAnalystAgent
    print("   ✅ DataAnalystAgent imported successfully")
    
    print("\n🎉 All imports successful!")
    
    # Test LLM initialization
    print("\n7. Testing LLM initialization...")
    llm = LLMHandler()
    print("   ✅ LLM initialized successfully")
    
    # Test database initialization  
    print("\n8. Testing database initialization...")
    db = DatabaseManager()
    print("   ✅ Database initialized successfully")
    
    print("\n✅ All tests passed! You can now run the main app.")
    
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("\n🔧 Fix: Make sure all required files are created in your project directory")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n🔧 Check your .env file and API keys")