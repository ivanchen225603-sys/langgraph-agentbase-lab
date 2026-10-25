"""Copy a packaged starter into a fresh directory: python -m agentbase.scaffold."""

import argparse
from importlib.resources import files
from pathlib import Path

from .contracts import AgentError

TEMPLATES = ("extraction", "research", "confirmed-action")


def scaffold(template: str, destination: Path) -> list[Path]:
    if template not in TEMPLATES:
        raise AgentError("scaffold_template", "unknown starter template")
    destination = Path(destination).absolute()
    # mkdir is atomic and refuses existing directories, including symlinks.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    root = files("agentbase").joinpath("starter_templates")
    copied = []
    for folder in (root.joinpath("shared"), root.joinpath(template)):
        for source in sorted(folder.iterdir(), key=lambda item: item.name):
            if not source.is_file():
                continue
            name = source.name.removesuffix(".template")
            target = destination / name
            with target.open("xb") as output:
                output.write(source.read_bytes())
            copied.append(target)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Create a professional Agent starter")
    parser.add_argument("template", choices=TEMPLATES)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        copied = scaffold(args.template, args.destination)
    except (AgentError, OSError) as exc:
        parser.exit(1, f"Scaffold failed: {type(exc).__name__}. Use a new writable directory.\n")
    print(f"Created {len(copied)} files in {args.destination.absolute()}")


if __name__ == "__main__":
    main()
