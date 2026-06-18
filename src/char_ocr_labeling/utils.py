def get_label_folder(label):
    """
    Map a single-character label to a standardized folder name.

    Rules:
        - Uppercase alphabets → "capital_<lowercase>"
        - Lowercase alphabets → "small_<char>"
        - Digits → unchanged
        - Other characters → "other_<char>"

    Args:
        label (str): Single-character label predicted by OCR

    Returns:
        str: Folder name for dataset organization

    Examples:
        'A' → 'capital_a'
        'b' → 'small_b'
        '5' → '5'
        '?' → 'other_?'
    """
    if label.isalpha():
        return f"capital_{label.lower()}" if label.isupper() else f"small_{label}"

    if label.isdigit():
        return label

    return f"other_{label}"