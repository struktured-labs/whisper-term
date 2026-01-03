# Voice to Claude

Push-to-talk voice input for Claude Code CLI using OpenAI Whisper.

## Usage

Hold **Super+V** to record, release to transcribe. Text is copied to clipboard by default.

### Modes
- `clipboard` - Copy transcription to clipboard (default)
- `type` - Type into active window
- `claude` - Type into active window + press Enter
- `stdout` - Print to stdout

```bash
./voice-to-claude.sh clipboard  # or type, claude, stdout
```

### Options
```bash
uv run python voice_to_claude.py --help
uv run python voice_to_claude.py --list-devices
uv run python voice_to_claude.py --model small.en  # more accurate
uv run python voice_to_claude.py --hotkey ctrl+shift+v
```

## Service

The systemd user service runs automatically on login:

```bash
systemctl --user status voice-to-claude
systemctl --user stop voice-to-claude
systemctl --user start voice-to-claude
```

## Requirements

- Python 3.10+
- PulseAudio/PipeWire
- xclip, xdotool (for clipboard/typing)
