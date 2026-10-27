set(UNI20_SANITIZE "" CACHE STRING
    "Enable sanitizers (comma- or semicolon-separated: address,undefined,thread,memory,leak)")
set_property(CACHE UNI20_SANITIZE PROPERTY TYPE STRING)

# Preserve the user's cache spelling; all consumers use the normalized list.
# Normalize even empty input so a parent directory's list cannot remain active.
string(REPLACE "," ";" UNI20_SANITIZE_LIST "${UNI20_SANITIZE}")
