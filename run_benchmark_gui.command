#!/bin/bash
cd "$(dirname "$0")"
python3 -m streamlit run research/app.py --server.headless true --browser.gatherUsageStats false
