#!/bin/sh
# commit-msg hook, wired through .pre-commit-config.yaml. Refuses a commit whose
# message credits an AI assistant as co-author: commits here are the author's.
if grep -iqE '^Co-Authored-By:.*(claude|anthropic|copilot|chatgpt|openai)' "$1"; then
  echo "commit-msg: remove the assistant Co-Authored-By trailer; commits are the author's." >&2
  exit 1
fi
