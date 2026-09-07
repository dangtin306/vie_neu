# -*- coding: utf-8 -*-
"""VieNeu-TTS v3 backend for the existing Flask server on port 8789.

This module keeps a lightweight controller in the parent process and runs the
long-lived VieNeu worker from test_vieneu_server.py in Conda environment 'tts_5'.
"""

import atexit
import json
import subprocess
import sys
from pathlib import Path
from threading import Lock

WORKER_SCRIPT = Path(__file__).resolve().with_name('test_vieneu_server.py')
PYTHON_TTS5 = Path('D:/hustmedia/conda_envs/tts_5/python.exe')
_worker = None
_worker_lock = Lock()
NL = chr(10)


def _start_worker():
    global _worker
    python_exe = PYTHON_TTS5 if PYTHON_TTS5.exists() else Path(sys.executable)
    _worker = subprocess.Popen(
        [str(python_exe), str(WORKER_SCRIPT), '--persistent-worker'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding='utf-8',
        bufsize=1,
    )
    message = None
    for _ in range(100):
        handshake = _worker.stdout.readline()
        if not handshake:
            break
        try:
            candidate = json.loads(handshake)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get('ready'):
            message = candidate
            break
        message = candidate
        break
    if message is None:
        raise RuntimeError('VieNeu v3 worker exited before ready')
    if not message.get('ready'):
        raise RuntimeError(message.get('message', 'VieNeu v3 worker failed'))


def main(text: str):
    """Run VieNeu v3 through the persistent worker and return its output path."""
    global _worker
    text = ''.join(char for char in str(text) if not 0xD800 <= ord(char) <= 0xDFFF)
    with _worker_lock:
        if _worker is None or _worker.poll() is not None:
            _start_worker()

        _worker.stdin.write(json.dumps({'text': text}, ensure_ascii=True) + NL)
        _worker.stdin.flush()
        response = _worker.stdout.readline()
        if not response:
            raise RuntimeError('VieNeu v3 worker disconnected')
        result = json.loads(response)
        if not result.get('ok'):
            raise RuntimeError(result.get('message', 'VieNeu v3 inference failed'))
    return Path(result.get('output_path', 'D:/hustmedia/python/tts/output/output_vieneu_persistent.mp3'))


def start():
    """Start the v3 worker during parent-server startup when enabled."""
    global _worker
    with _worker_lock:
        if _worker is None or _worker.poll() is not None:
            _start_worker()
    print('VieNeu v3 worker started and ready.')


def stop():
    """Stop the worker and release all RAM/VRAM when disabled."""
    global _worker
    with _worker_lock:
        if _worker is not None and _worker.poll() is None:
            try:
                _worker.stdin.write(json.dumps({'stop': True}) + NL)
                _worker.stdin.flush()
                _worker.wait(timeout=5)
            except Exception:
                try:
                    _worker.kill()
                except Exception:
                    pass
        _worker = None
        print('VieNeu v3 worker stopped and released.')


atexit.register(stop)
