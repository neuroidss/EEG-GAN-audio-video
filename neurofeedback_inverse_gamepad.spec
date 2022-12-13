# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None


a = Analysis(
    ['neurofeedback_inverse_gamepad.py'],
    pathex=[],
    binaries=[("/content/env_pyinstaller1/lib/python3.9/site-packages/_libsuinput.cpython-39-x86_64-linux-gnu.so", "."),
    ("/content/env_pyinstaller1/lib/python3.9/site-packages/brainflow/lib/libBoardController.so", "brainflow/lib"),
    ("/content/env_pyinstaller1/lib/python3.9/site-packages/pylsl/lib/liblsl.so", "pylsl/lib"),
    ],
    datas=[("/content/env_pyinstaller1/lib/python3.9/site-packages/mne", "mne")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='neurofeedback_inverse_gamepad',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
