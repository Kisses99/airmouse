# Lumina Air Mouse

A high-performance "Air Mouse" using Python, MediaPipe, and OpenCV. Control your computer cursor and perform gestures using just your webcam.

## Features
- **Precise Cursor Control**: Smooth index finger tracking.
- **Dwell Clicking**: Hover over an area to trigger a left click.
- **Drag Gestures**: Pinch (Index + Thumb) for left drag, Fist for right drag.
- **Scroll Support**: Two-finger vertical movement for scrolling.
- **Visual Feedback**: Real-time HUD showing tracking status and gesture progress.

## Installation

### Using pip
```bash
pip install airmouse
```

### From Source (Development)
1. Clone the repository:
   ```bash
   git clone https://github.com/Kisses99/airmouse.git
   cd airmouse
   ```
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

## Usage

Run the air mouse via the command line:

```bash
airmouse
```

Or if running from source:
```bash
uv run airmouse
```

## Gestures
- **Move Cursor**: Point with your index finger.
- **Left Click**: Hold the cursor still on a point for 0.8s.
- **Left Drag**: Pinch your index finger and thumb together and move.
- **Right Drag**: Make a fist and move.
- **Scroll**: Hold your index and middle fingers up together and move up/down.

## Requirements
- Python 3.10 or higher
- A webcam
- Windows OS (Currently optimized for Windows via `DirectShow`)

## License
MIT
