#!/usr/bin/env bash

# grab current directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# enable virtual environment at .venv
source "$DIR/.venv/bin/activate"

# run main.py with the provided directory path
python "$DIR/main.py" "$@"
