ai agent that scrapes online marketplaces for specified products
want agent to be as autonomous as possible with mcp (for modularity/compatibility)and minimal hard-coded functionalities
first, will provide overarching plan. then will select next immediate action. validate against set list of possible actions (limit potentially malicious actions). then will execute the next action. the result will be returned to the agent. repeat.

capping domains to ebay and facebook marketplace for now -- parallelizing and generalizable to others

virtual env setup
create venv
python -m venv .venv

activate venv
.venv\Scripts\activate

run the program
uv run main.py

stack

- uv package manager
- gemini api
- playwright mcp
- postgres db mcp
- email mcp (write myself)

other anthropic models https://docs.anthropic.com/en/docs/about-claude/model-deprecations
