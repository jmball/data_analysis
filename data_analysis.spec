# -*- mode: python ; coding: utf-8 -*-

import pathlib
import site

import gooey

gooey_root = os.path.dirname(gooey.__file__)
for path in site.getsitepackages():
    if path.endswith("site-packages"):
        site_pkgs = pathlib.Path(path)
        break
pptx_templates = site_pkgs.joinpath('pptx').joinpath('templates').joinpath('default.pptx')

cwd = pathlib.Path.cwd()
filename = "data_analysis.py"

block_cipher = None

a = Analysis([filename],
             pathex=[str(cwd.joinpath(filename))],
             binaries=[],
             datas=[(str(pptx_templates), str(pathlib.Path('.').joinpath('pptx').joinpath('templates')))],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name=filename.strip(".py"),
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon=os.path.join(gooey_root, 'images', 'program_icon.ico'))
