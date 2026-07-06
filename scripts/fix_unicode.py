"""Fix non-ASCII chars in test_streaming.py for Windows cp1252 terminals."""
import pathlib, re

target = pathlib.Path(__file__).parent / "test_streaming.py"
text = target.read_text(encoding="utf-8")

SUBS = {
    "\u2550": "=", "\u2500": "-", "\u2502": "|",
    "\u251c": "+", "\u2514": "+", "\u2518": "+", "\u2510": "+", "\u250c": "+",
    "\u2713": "[PASS]", "\u2717": "[FAIL]", "\u2714": "[PASS]", "\u2718": "[FAIL]",
    "\u27a4": ">", "\u2192": "->", "\u2190": "<-",
    "\u2014": "--", "\u2013": "-",
    "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"',
    "\U0001f30a": "",
    "\u26a0": "!",
    "\u2714": "[OK]",
    "\u2022": "*",
    "\u25b6": ">",
    "\u25cf": "*",
}

for bad, good in SUBS.items():
    text = text.replace(bad, good)

remaining = [(i + 1, repr(c)) for i, c in enumerate(text) if ord(c) > 127]
if remaining:
    print(f"Still {len(remaining)} non-ASCII chars remaining:")
    for line_context, char in remaining[:10]:
        print(f"  pos {line_context}: {char}")
else:
    print("All non-ASCII characters removed.")

target.write_text(text, encoding="utf-8")
print(f"Saved: {target}")
