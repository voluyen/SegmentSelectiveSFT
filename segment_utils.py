"""Chia solution thanh segment - dung chung cho ca Attribution/ va SelectiveSFT/.

Truoc day pattern regex bi copy vao segment_split.py va train_mask.py; hai ban
lech nhau la selected_spans_ids tro nham doan text ma khong bao loi. Gio ca hai
import tu day.

Bat buoc: "".join(split_segments(t)) == t  va khong co segment rong.
"""
import re

# Cach chia cua paper: cat tai cac tu khoa backtracking.
CUE_PATTERN = r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)"
# Cach chia theo doan van: moi "\n\n" mo mot segment moi.
PARAGRAPH_PATTERN = r"(\n\n)"

SEGMENT_PATTERNS = {
    "cue": CUE_PATTERN,
    "paragraph": PARAGRAPH_PATTERN,
}
DEFAULT_MODE = "paragraph"


def split_segments(text, mode=DEFAULT_MODE):
    """Chia text thanh list segment, delimiter dinh vao DAU segment sau no.

    Ghep lai bang "".join(...) phai ra dung text ban dau - grad_analyze.py tinh
    span token bang do dai cac prefix cong don nen khong duoc mat ky tu nao.
    """
    if mode not in SEGMENT_PATTERNS:
        raise ValueError("mode phai la mot trong %s, nhan duoc %r" % (sorted(SEGMENT_PATTERNS), mode))
    parts = re.split(SEGMENT_PATTERNS[mode], text)

    segments = [parts[0]]
    for j in range(1, len(parts), 2):
        segments.append(parts[j] + parts[j + 1])

    return _drop_empty(segments, text)


def _drop_empty(segments, text):
    """Gop segment rong vao segment ben canh.

    Chia theo "\\n\\n" de lai chuoi rong khi text bat dau bang "\\n\\n" hoac co
    tu 4 newline lien tiep tro len. Segment rong = 0 token, lam
    get_important_segments.py chia cho 0 (sum|IG| / len**0.5) va ra nan.
    """
    merged = []
    for seg in segments:
        if seg == "":
            continue
        if merged and merged[-1] == "":
            merged[-1] = seg
        else:
            merged.append(seg)

    if not merged:
        # Ca text la chuoi rong: tra ve nguyen van de assert join van dung.
        return [text]

    assert "".join(merged) == text, "split_segments lam mat ky tu"
    return merged
