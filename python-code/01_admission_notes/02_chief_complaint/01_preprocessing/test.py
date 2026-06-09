from quickumls import QuickUMLS

matcher = QuickUMLS("QUICKUMLS_INDEX_DIR")

text = "chest pain and shortness of breath"
matches = matcher.match(text, best_match=True, ignore_syntax=False)

print(matches)