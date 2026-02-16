#!/usr/bin/env python3
"""
Verification script for extracted infrastructure modules.
This tests the new infra/ and schemas/ modules without importing the old Flask code.
"""
from __future__ import annotations

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auth_module():
    """Test auth module functions."""
    print("Testing auth module...")
    
    # Temporarily remove chatmock from sys.modules to avoid __init__.py imports
    chatmock_backup = sys.modules.pop('chatmock', None)
    
    try:
        # We need to mock chatmock.config since it's not converted yet
        import types
        config = types.ModuleType('chatmock.config')
        config.CLIENT_ID_DEFAULT = 'test-client-id'
        config.OAUTH_TOKEN_URL = 'https://auth.openai.com/oauth/token'
        sys.modules['chatmock.config'] = config
        
        # Mock chatmock.core.models with actual dataclass
        from dataclasses import dataclass
        
        @dataclass
        class PkceCodes:
            code_verifier: str
            code_challenge: str
        
        models = types.ModuleType('chatmock.core.models')
        models.PkceCodes = PkceCodes
        sys.modules['chatmock.core.models'] = models
        
        # Now we can import auth
        from chatmock.infra.auth import parse_jwt_claims, generate_pkce, eprint
        from chatmock.infra.auth import read_auth_file, write_auth_file
        
        pkce = generate_pkce()
        assert len(pkce.code_verifier) > 0, "code_verifier should not be empty"
        assert len(pkce.code_challenge) > 0, "code_challenge should not be empty"
        print('✓ AUTH OK')
    finally:
        # Restore chatmock module
        if chatmock_backup:
            sys.modules['chatmock'] = chatmock_backup


def test_sse_module():
    """Test SSE async generators."""
    print("Testing SSE module...")
    
    from chatmock.infra.sse import sse_translate_chat, sse_translate_text
    import inspect
    
    assert inspect.isasyncgenfunction(sse_translate_chat), "sse_translate_chat should be async generator"
    assert inspect.isasyncgenfunction(sse_translate_text), "sse_translate_text should be async generator"
    print('✓ SSE ASYNC OK')


def test_messages_module():
    """Test message conversion."""
    print("Testing messages module...")
    
    from chatmock.schemas.messages import convert_chat_messages_to_responses_input
    
    result = convert_chat_messages_to_responses_input([
        {'role': 'user', 'content': 'hello'}
    ])
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert result[0]['type'] == 'message', f"Expected type 'message', got {result[0]['type']}"
    assert result[0]['role'] == 'user', f"Expected role 'user', got {result[0]['role']}"
    print('✓ MESSAGES OK')


def test_all_imports():
    """Test that all modules can be imported."""
    print("Testing all imports...")
    
    from chatmock.infra.session import ensure_session_id
    from chatmock.infra.limits import parse_rate_limit_headers
    from chatmock.schemas.transform import convert_ollama_messages
    
    print('✓ ALL IMPORTS OK')


if __name__ == '__main__':
    try:
        test_auth_module()
        test_sse_module()
        test_messages_module()
        test_all_imports()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
