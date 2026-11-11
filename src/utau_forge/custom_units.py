"""User labels are separate from internal identities and safe WAV stems."""
import re
from hashlib import sha256


def default_stem(label: str) -> str:
    if label.isascii():
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_-")[:60]
        if stem and stem.split("_")[0].upper() not in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
            return stem
    return "custom_" + sha256(label.encode("utf-8")).hexdigest()[:10]


def validate_unit(label: str, stem: str | None = None) -> tuple[str, str]:
    label = label.strip()
    if not label or len(label) > 64 or any(ord(char) < 32 or char in ",=" for char in label):
        raise ValueError("标签需要 1–64 个字符，不能包含逗号、等号或控制字符")
    try:
        label.encode("cp932")
    except UnicodeEncodeError as error:
        raise ValueError("标签含有 Shift-JIS 音源不支持的字符，请改用可导出的文字或罗马音标签") from error
    stem = (stem or "").strip() or default_stem(label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", stem):
        raise ValueError("文件名前缀请使用 1–60 个英文字母、数字、下划线或连字符，以字母或数字开头")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", stem, re.IGNORECASE):
        raise ValueError("文件名前缀不能使用 Windows 保留名称")
    return label, stem
