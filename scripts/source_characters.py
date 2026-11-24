"""Count Nasubi extractor source from a source checkout."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/openif/architectures/gpr_1_nasubi"
FILES = ("extractor.py", "projection.py", "rules.py", "morphology.py")


def main():
    counts = {}
    for file in FILES:
        source = (ROOT / file).read_bytes().decode("utf-8")
        counts[file] = {
            "characters": len(source),
            "lines": len(source.splitlines()),
            "if_elif": sum(
                isinstance(node, ast.If) for node in ast.walk(ast.parse(source))
            ),
        }
    lexicon = (ROOT / "lexicon.json").read_bytes().decode("utf-8")
    rules = json.loads(lexicon)
    print(
        json.dumps(
            {
                "files": counts,
                "total": {
                    metric: sum(row[metric] for row in counts.values())
                    for metric in ("characters", "lines", "if_elif")
                },
                "lexicon": {
                    "characters": len(lexicon),
                    "lines": len(lexicon.splitlines()),
                    "rules": len(rules),
                    "entries": sum(
                        len(words) for rule in rules.values() for words in rule.values()
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
