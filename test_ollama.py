#!/usr/bin/env python3
"""
Test script for Ollama Manager
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model import OllamaManager

def test_ollama_manager():
    """Test basic Ollama manager functionality."""
    print("🧪 Testing Ollama Manager...")

    with OllamaManager() as manager:
        # Test binary check
        print("1. Testing Ollama binary check...")
        has_binary = manager._check_ollama_binary()
        print(f"   Binary available: {has_binary}")

        if not has_binary:
            print("❌ Ollama binary not available, skipping further tests")
            return False

        # Test server start (without pulling models for this test)
        print("2. Testing server initialization...")
        server_started = manager._start_ollama_server()
        print(f"   Server started: {server_started}")

        if server_started:
            print("3. Testing model status check...")
            # Just check status without pulling
            for model in ["mistral", "nomic-embed-text"]:
                status = manager._check_model_status(model)
                print(f"   {model}: {status.status.value}")

            print("4. Generating status report...")
            report = manager.get_model_status_report()
            print(report)

        print("✅ Ollama Manager test completed")
        return True

if __name__ == "__main__":
    try:
        test_ollama_manager()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
