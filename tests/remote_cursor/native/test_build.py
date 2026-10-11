"""Portable recipe tests; all Windows subprocesses are mocked, never executed."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("probe_build", Path(__file__).with_name("build.py"))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BuildRecipeTests(unittest.TestCase):
    def run_recipe(self, arguments=(), native_results=(0, 0), compile_result=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vswhere = root / "program files/Microsoft Visual Studio/Installer/vswhere.exe"
            devcmd = root / "visual studio/Common7/Tools/VsDevCmd.bat"
            for path in (vswhere, devcmd):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            calls, results = [], iter(native_results)

            def run(command, **kwargs):
                calls.append((command, kwargs))
                if command[0] == str(vswhere):
                    return SimpleNamespace(stdout=str(root / "visual studio") + "\n", returncode=0)
                if command[0] == "cmd.exe":
                    return SimpleNamespace(returncode=compile_result)
                result = next(results)
                if isinstance(result, Exception):
                    raise result
                return SimpleNamespace(returncode=result)

            with patch.object(build, "__file__", str(root / "build.py")), \
                 patch.object(build, "os", SimpleNamespace(name="nt", environ={"ProgramFiles(x86)": str(root / "program files")})), \
                 patch.object(build.sys, "getwindowsversion", return_value="mocked Windows version", create=True), \
                 patch.object(build.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                result = build.main(list(arguments))
            return result, calls, (root / "out/build.cmd").read_text(encoding="utf-8")

    def test_default_compiles_without_running_probe(self):
        result, calls, batch = self.run_recipe()
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn('/std:c++20', batch)
        self.assertIn('/W4 /WX', batch)
        self.assertIn('user32.lib magnification.lib', batch)
        self.assertNotIn('powershell', batch)

    def test_explicit_opt_in_runs_both_backends(self):
        result, calls, _ = self.run_recipe(["--run-isolated-desktop"])
        self.assertEqual(result, 0)
        self.assertEqual([c[0][-1] for c in calls[2:]], ['--backend=user32', '--backend=magnification'])
        self.assertTrue(all('--initial-global-visible' in c[0] for c in calls[2:]))
        self.assertTrue(all(c[1]['timeout'] == 75 for c in calls[2:]))

    def test_unavailable_is_not_pass_and_other_backend_runs(self):
        result, calls, _ = self.run_recipe(["--run-isolated-desktop"], (2, 0))
        self.assertEqual((result, len(calls)), (2, 4))

    def test_cleanup_failure_prevents_second_backend(self):
        result, calls, _ = self.run_recipe(["--run-isolated-desktop"], (4,))
        self.assertEqual((result, len(calls)), (4, 3))

    def test_timeout_prevents_second_backend(self):
        result, calls, _ = self.run_recipe(["--run-isolated-desktop"], (subprocess.TimeoutExpired('probe', 75),))
        self.assertEqual((result, len(calls)), (4, 3))

    def test_failed_build_never_runs_probe(self):
        result, calls, _ = self.run_recipe(["--run-isolated-desktop"], compile_result=1)
        self.assertEqual((result, len(calls)), (1, 2))

    def test_non_windows_is_explicitly_unavailable(self):
        with patch.object(build, "os", SimpleNamespace(name="posix")), \
             patch.object(build.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build.main([]), 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
