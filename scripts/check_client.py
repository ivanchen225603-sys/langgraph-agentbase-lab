"""Check the actual shipped JS modules with Node (development dependency)."""

import shutil
import subprocess


def main():
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js >=22 is required for client verification")
    version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        raise RuntimeError("Node.js >=22 is required for client verification")
    for args in [
        ("--check", "raglab/web/agent-ui.js"),
        ("--check", "raglab/web/cancel-status.mjs"),
        ("--check", "raglab/web/intent-ui.mjs"),
        (
            "--test",
            "tests/test_sse_decoder.mjs",
            "tests/test_agent_ui.mjs",
            "tests/test_agent_ui_cancel_observation.mjs",
            "tests/test_agent_ui_closure.mjs",
            "tests/test_cancel_status.mjs",
            "tests/test_intent_ui.mjs",
        ),
    ]:
        result = subprocess.run([node, *args], timeout=30, check=False)
        if result.returncode:
            return result.returncode
    print(f"client modules checked with {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
