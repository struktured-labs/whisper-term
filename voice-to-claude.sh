#!/bin/bash
# Voice-to-Claude launcher
# Run with: ./voice-to-claude.sh [mode]
# Modes: clipboard (default), type, claude, stdout

cd "$(dirname "$0")"

# Source setenv if exists
[[ -f setenv.sh ]] && source setenv.sh

MODE="${1:-clipboard}"
MODEL="${WHISPER_MODEL:-base.en}"
DEVICE="${WHISPER_DEVICE:-7}"  # 7=pulse, uses system default (M4)

exec uv run python voice_to_claude.py --mode "$MODE" --model "$MODEL" --device "$DEVICE"
