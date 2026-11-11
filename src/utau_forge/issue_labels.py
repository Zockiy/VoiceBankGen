"""Chinese explanations for stored machine-readable quality flags."""
ISSUE_LABELS = {
    "ZH_TIMING_REVIEW": "普通话短音：初始标线需试听校对",
    "CROSS_LANGUAGE_APPROXIMATE": "跨语种近似：发音差异需试听校对",
    "NO_SUSTAIN_REGION": "未找到稳定元音区",
    "DEVOICED_VOWEL": "元音清化，难以延长",
    "SPECIAL_UNIT_REVIEW_REQUIRED": "特殊音节需手动校对",
    "STALE_EDIT": "标线已过期，需重新校对",
    "STALE_ALIGNMENT": "对齐结果已更新，需重新选音",
    "TIMING_PENDING": "尚未完成标线分析",
    "REPAIR_REVIEW_REQUIRED": "拼接或补尾音需试听确认",
    "COARTICULATION_UNCERTAIN": "拼接衔接不确定，需试听",
    "ALIGNMENT_FAILED": "音素对齐失败",
    "ASR_EMPTY": "未识别出语音文本",
    "ASR_SPEECH_UNCERTAIN": "语音识别可信度较低",
    "LONG_SPEECH_SPLIT_LOW_ENERGY": "长语句已在低音量处切分",
    "LONG_SPEECH_UNSPLIT": "长语句未能自动切分",
}


def issue_text(flags, fallback="边界可用"):
    # Keep unrecognized internal codes in the tooltip, not in the main column.
    return "；".join(ISSUE_LABELS.get(flag, "其他问题，查看悬停详情") for flag in flags) if flags else fallback


def issue_details(flags):
    return "\n".join(f"{ISSUE_LABELS.get(flag, '其他问题')}（{flag}）" for flag in flags)
