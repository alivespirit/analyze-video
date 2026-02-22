import re
from datetime import datetime
from typing import Optional

OLD_FOLDER_RE = re.compile(r"^(?P<date>\d{8})(?P<hour>\d{2})$")
OLD_NAME_MMSS_RE = re.compile(r"^(?P<mm>\d{2})M(?P<ss>\d{2})S")
OLD_NAME_DIGITS_RE = re.compile(r"^(?P<mm>\d{2})(?P<ss>\d{2})")
NEW_NAME_TS14_RE = re.compile(r"(?P<ts>\d{14})")


def parse_datetime_from_path(video_path: str) -> Optional[datetime]:
    """Parse datetime from either legacy or new camera path formats.

    Legacy: "YYYYMMDDHH/minute-second prefix" e.g., 2026021408/06M45S_....mp4
    New:    "..._YYYYMMDDHHMMSS.mp4" e.g., 2026/02/05/Balcony_00_20260205174608.mp4
    """
    try:
        parts = re.split(r"[\\/]", video_path)
        if not parts:
            return None
        basename = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else ""
        # Try new format: 14-digit timestamp anywhere in basename
        m_new = NEW_NAME_TS14_RE.search(basename)
        if m_new:
            ts = m_new.group('ts')
            try:
                return datetime.strptime(ts, "%Y%m%d%H%M%S")
            except ValueError:
                pass
        # Try legacy folder + mm/ss from basename
        m_folder = OLD_FOLDER_RE.match(parent)
        if m_folder:
            date = m_folder.group('date')  # YYYYMMDD
            hour = m_folder.group('hour')  # HH
            mm = None
            ss = None
            m_mmss = OLD_NAME_MMSS_RE.match(basename)
            if m_mmss:
                mm = m_mmss.group('mm')
                ss = m_mmss.group('ss')
            else:
                m_digits = OLD_NAME_DIGITS_RE.match(basename)
                if m_digits:
                    mm = m_digits.group('mm')
                    ss = m_digits.group('ss')
            if mm and ss:
                try:
                    return datetime.strptime(f"{date}{hour}{mm}{ss}", "%Y%m%d%H%M%S")
                except ValueError:
                    return None
        return None
    except Exception:
        return None


def format_timestamp_for_caption(dt: datetime) -> str:
    """Return legacy-style caption prefix using hour and minute only.
    Example: "_08H06M:_ "
    """
    return f"_{dt.strftime('%H')}H{dt.strftime('%M')}M:_ "


def hhmm_from_video_path(video_path: str) -> Optional[str]:
    dt = parse_datetime_from_path(video_path)
    return dt.strftime('%H:%M') if dt else None
