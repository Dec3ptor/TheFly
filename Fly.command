#!/usr/bin/env bash
# Double-click this in Finder to run the simulator. macOS opens it in Terminal.
cd "$(dirname "${BASH_SOURCE[0]}")"
./fly
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "Exited with status $status — leaving this window open so you can read it."
  read -r -p "Press return to close."
fi
