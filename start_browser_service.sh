#!/bin/bash
python parlai/chat_service/services/browser_chat/run.py    --config-path  parlai/chat_service/tasks/chatbot/config.yml --port 1234  >/dev/null 2>&1 &
sleep 15
python parlai/chat_service/services/browser_chat/client.py  --port 1234