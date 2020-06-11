#!/bin/bash
python parlai/chat_service/services/browser_chat/run.py --config-path  parlai/chat_service/tasks/chatbot/config.yml --port 9924 &
sleep 20
python parlai/chat_service/services/browser_chat/client.py --port 9924
