# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('google')


a = Analysis(
    ['C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-NoiseProcessing\\NoiseProcessing\\YAWN_GUI.py'],
    pathex=['C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-NoiseProcessing\\NoiseProcessing'],
    binaries=[],
    datas=[('C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-NoiseProcessing\\NoiseProcessing\\ExampleApplications\\Figures', 'ExampleApplications/Figures')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='YAWN_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
