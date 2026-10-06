import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('quiet', [False, True])
def test_dependency_filter_is_opt_in_specific_and_inherited(quiet):
    emit = """
import hipct_seg_debug
import warnings
warnings.warn_explicit('TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES',
                      UserWarning, 'dependency.py', 1, module='paramiko.pkey')
warnings.warn_explicit('Blowfish has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.Blowfish',
                      UserWarning, 'dependency.py', 2, module='paramiko.transport')
warnings.warn_explicit('Other dependency problem', UserWarning, 'dependency.py', 3, module='paramiko.transport')
warnings.warn_explicit('Geometry warning', RuntimeWarning, 'geometry.py', 4, module='hipct_seg_debug.edit.centreline_refine')
"""
    code = emit + '\nimport subprocess, sys\nsubprocess.run([sys.executable, "-c", '+repr(emit)+'], check=True)'
    env = os.environ.copy()
    env.pop('HIPCT_QUIET_DEPENDENCY_WARNINGS', None)
    env['PYTHONWARNINGS'] = 'default'
    command = [sys.executable, '-c', code]
    if quiet:
        command.append('--quiet-dependency-warnings')
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    assert ('TripleDES has been moved' in result.stderr) is not quiet
    assert ('Blowfish has been moved' in result.stderr) is not quiet
    assert result.stderr.count('Other dependency problem') == 2
    assert result.stderr.count('Geometry warning') == 2
