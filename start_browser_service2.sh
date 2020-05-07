#!/bin/bash
python parlai/chat_service/services/browser_chat2/run.py    --config-path  parlai/chat_service/tasks/chatbot/config2.yml --port 1678 &
sleep 20
python parlai/chat_service/services/browser_chat2/client.py  --port 1678