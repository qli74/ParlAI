#!/bin/bash
python parlai/chat_service/services/browser_chat/run.py    --config-path  parlai/chat_service/tasks/chatbot/config.yml >/dev/null 2>&1 &
sleep 6
python parlai/chat_service/services/browser_chat/client.py