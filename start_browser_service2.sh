#!/bin/bash
python parlai/chat_service/services/browser_chat2/run.py    --config-path  parlai/chat_service/tasks/chatbot/config2.yml --port 5678 &
sleep 10
python parlai/chat_service/services/browser_chat2/client.py  --port 5678