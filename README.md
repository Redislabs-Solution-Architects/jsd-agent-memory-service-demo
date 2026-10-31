# jsd-agent-memory-service-demo
Support Chat Bot with leverages the following Redis Iris Services: 
- Agent Memory Service -  to maintain session state and long term memory extraction and retreival.
- Lang Cache - a repository of FAQs which can be retreived by avoiding a call to LLMs
- RedisVL - Python SDK which provides Intent Routing to route requests 3 ways - LLM, Lang Cache, Trash

## Requirements

- Redis Cloud DB Subscription (Essentials or Higher)
- Redis Agent Memory Service
- Redis Lang Cache Service
- OpenAI API Key
