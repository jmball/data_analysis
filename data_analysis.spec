# -*- mode: python -*-

import os
import site
import sys

import gooey

# get paths to gooey files
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix = 'gooey/images')

# get paths to required data and dlls
site_pkgs = site.getsitepackages()[1]
pptx_templates = os.path.join(site_pkgs, r'pptx\templates\default.pptx')
scipy_dlls = os.path.join(site_pkgs, r'scipy\extra-dll')

sys.modules['FixTk'] = None

block_cipher = None

a = Analysis(['data_analysis.py'],
             datas=[(pptx_templates, '.\\pptx\\templates\\')],
             pathex=[scipy_dlls],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=['pyqt',
                       'PyQt5',
                       'lib2to3',
                       'FixTk',
                       'tcl',
                       'tk',
                       '_tkinter',
                       'tkinter',
                       'Tkinter',
                       'zmq',
                       'pyzmq',
                       'IPython']
             )

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

options = [('u', None, 'OPTION')]

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          options,
          gooey_languages,
          gooey_images,
          name='data_analysis',
          debug=False,
          strip=False,
          upx=True,
          console=False)

# coll = COLLECT(exe,
#                a.binaries,
#                a.zipfiles,
#                a.datas,
#                gooey_languages,
#                gooey_images,
#                strip=False,
#                upx=True,
#                name='data_analysis')
