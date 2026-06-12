import os
from pathlib import Path

from quickumls import QuickUMLS

quickumls_index_dir = Path(
    os.environ.get(
        "QUICKUMLS_INDEX_DIR",
        Path.home() / "Downloads" / "thesis" / "quickumls_index_2026AA",
    )
)
matcher = QuickUMLS(str(quickumls_index_dir))

text = "chest pain and shortness of breath"
matches = matcher.match(text, best_match=True, ignore_syntax=False)

print(matches)
