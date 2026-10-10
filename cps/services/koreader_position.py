# -*- coding: utf-8 -*-
# Calibre-Web Automated – fork of Calibre-Web
# Copyright (C) 2024-2026 Calibre-Web-NextGen contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# See CONTRIBUTORS for full list of authors.
"""Exact reading positions between KOReader, the web reader and a Kobo (#324).

KOReader reports its place as a crengine XPointer into the file it holds; the
web reader reports an epub.js CFI into the EPUB it renders. Handing only the
percentage across lands each reader pages away from the other, so this module
converts the position itself (``koreader_xpointer``) -- but only when the
conversion is provably about the same file on both sides:

* the file is identified the way KOReader identifies it, by the partial MD5 of
  its bytes (``checksums/koreader.py``): a device's document digest names the
  file it holds, and the library EPUB's own digest (cached per path, size and
  modification time) names the file the web reader renders;
* a KOReader report is journalled per device with the digest it was sent under
  (``journal_report``), because the shared ``kosync_progress`` row is keyed by
  book id and no longer says which file its XPointer belongs to.

A Kobo's place reaches KOReader the same way (``kobo_position_for_device``):
its ``KoboSpan`` addresses the KEPUB the Kobo holds, and crosses into the
library EPUB through the text the two files share (``kepub_alignment``), when
the span is provably the Kobo's own latest report behind the row and the Kobo
provably holds the library KEPUB as it is now.

The other way (``kobo_location_for_device``), a KOReader or web-reader place
is handed to a Kobo as a span of the library KEPUB when it is served: the
shared bookmark's own ``Location`` is the Kobo's last report, which Nickel
would seek to whatever the percentage says. The place crosses exactly when
that reader's file is provably the library EPUB, and otherwise as the span at
the shared percentage; either only to a Kobo that holds the library KEPUB.

Anything short of that proof leaves the hand-off at the percentage, as before.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import unquote

from sqlalchemy import func

from .. import logger

log = logger.create()

# ``DeviceReadingPosition.location_type`` of a KOReader report. Its
# ``location_value`` is the XPointer and its ``location_source`` the partial MD5
# of the file that XPointer addresses (the digest the device sent it under).
KOREADER_LOCATION_TYPE = "koreader_xpointer"

# The ``kosync_progress.device`` of a row a Kobo's sync wrote
# (``kobo.share_kobo_progress_with_koreader``).
KOBO_DEVICE = "Kobo"

_MAX_XPOINTER_CHARS = 4096
_SHA256_MAX = 64
_SHA256 = OrderedDict()
_SHA256_LOCK = threading.Lock()


def is_xpointer(progress) -> bool:
    """True for a crengine XPointer (a reflowable-document position).

    KOReader sends a page number instead for fixed-layout documents (PDF,
    CBZ), and the web reader's sentinel is neither.
    """
    return (isinstance(progress, str) and progress.startswith("/body/")
            and len(progress) <= _MAX_XPOINTER_CHARS)


def file_digest(epub_path) -> Optional[str]:
    """KOReader's partial MD5 of a file (lower case), or ``None``."""
    from .koreader_library import file_facts
    try:
        _size, checksum = file_facts(os.fspath(epub_path))
    except (OSError, TypeError):
        return None
    return checksum.lower() if checksum else None


def _identity(path):
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sha256(path, identity) -> str:
    """SHA-256 of the whole file, cached per file identity."""
    key = (path,) + identity
    with _SHA256_LOCK:
        cached = _SHA256.get(key)
        if cached is not None:
            _SHA256.move_to_end(key)
            return cached
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    with _SHA256_LOCK:
        _SHA256[key] = value
        while len(_SHA256) > _SHA256_MAX:
            _SHA256.popitem(last=False)
    return value


def web_resume(epub_path, xpointer: str, device_digests: Iterable[str]) -> Optional[dict]:
    """``{"cfi", "epub_sha256"}`` for a KOReader XPointer, or ``None``.

    ``device_digests`` are the partial MD5s the XPointer was reported under;
    one of them must be the digest of ``epub_path`` itself, or the XPointer
    addresses some other file and is not converted. The SHA-256 lets the web
    reader confirm it renders these same bytes (``readerResume.ts``).
    """
    if not is_xpointer(xpointer):
        return None
    from .koreader_xpointer import xpointer_to_cfi
    try:
        path = os.fspath(epub_path)
        before = _identity(path)
        wanted = {d.lower() for d in device_digests if isinstance(d, str) and d}
        if not wanted or file_digest(path) not in wanted:
            return None
        cfi = xpointer_to_cfi(path, xpointer)
        if cfi is None:
            return None
        fingerprint = _sha256(path, before)
        if _identity(path) != before:
            return None  # replaced while we read it
    except (OSError, TypeError):
        return None
    return {"cfi": cfi, "epub_sha256": fingerprint}


def device_xpointer(epub_path, cfi: str, device_digest: str) -> Optional[str]:
    """The XPointer of a web-reader CFI in the file a device holds, or ``None``.

    Only when ``device_digest`` (the document digest the device asked with) is
    the digest of ``epub_path``, the EPUB the web reader renders: an XPointer
    computed from any other file is never handed to a device.
    """
    if not isinstance(device_digest, str) or not device_digest:
        return None
    from .koreader_xpointer import cfi_to_xpointer
    try:
        path = os.fspath(epub_path)
        before = _identity(path)
        if file_digest(path) != device_digest.lower():
            return None
        xpointer = cfi_to_xpointer(path, cfi)
        if xpointer is None or _identity(path) != before:
            return None
    except (OSError, TypeError):
        return None
    return xpointer


def _web_reader_cfi(user_id, book_id, percentage) -> Optional[str]:
    """The EPUB CFI the web reader saved at exactly ``percentage``, or ``None``.

    The user's saved EPUB bookmark qualifies only when the browser journal
    holds that same CFI at that same percentage, so it is provably the place
    the shared percentage was made from (the bookmark row is per format, which
    also rules out a KEPUB CFI from the classic reader).
    """
    from .. import ub

    cfi = ub.session.query(ub.Bookmark.bookmark_key).filter(
        ub.Bookmark.user_id == int(user_id),
        ub.Bookmark.book_id == int(book_id),
        ub.Bookmark.format == "epub",
    ).scalar()
    if not cfi:
        return None
    reported = ub.session.query(ub.DeviceReadingPosition.id).join(
        ub.Device, ub.Device.id == ub.DeviceReadingPosition.device_id,
    ).filter(
        ub.Device.user_id == int(user_id),
        ub.Device.kind == "webreader",
        ub.DeviceReadingPosition.book_id == int(book_id),
        ub.DeviceReadingPosition.cfi == cfi,
        ub.DeviceReadingPosition.progress_percent == percentage,
    ).first()
    return cfi if reported is not None else None


def web_position_for_device(*, user_id, book_id, record, document) -> Optional[str]:
    """The XPointer, in the requesting device's file, of the web reader's place.

    ``record`` is the ``kosync_progress`` row about to be served. It qualifies
    only when the web reader wrote it (a percentage-only row) and the CFI that
    produced it is known for certain: the browser journal holds an EPUB CFI at
    exactly that percentage which is also the user's saved EPUB bookmark (the
    bookmark row is per format, so this also rules out a KEPUB CFI from the
    classic reader). The XPointer is then derived per request against the
    library EPUB, and only if ``document`` -- the digest the device asked with
    -- is that EPUB's digest (``device_xpointer``).
    """
    from .. import calibre_db
    from ..annotations import _book_format_path
    from ..progress_syncing.protocols.kosync import (PERCENTAGE_ONLY_LOCATOR,
                                                     WEB_READER_DEVICE)
    from .parallel import run_blocking

    if (record is None or record.progress != PERCENTAGE_ONLY_LOCATOR
            or record.device != WEB_READER_DEVICE or record.percentage is None):
        return None
    cfi = _web_reader_cfi(user_id, book_id, record.percentage)
    if not cfi:
        return None
    book = calibre_db.get_book(int(book_id))
    epub_path = _book_format_path(book, "EPUB") if book is not None else None
    if not epub_path:
        return None
    return run_blocking(lambda: device_xpointer(epub_path, cfi, document))


def kobo_device_xpointer(epub_path, kepub_path, source: str, span_id: str,
                         device_digest: str) -> Optional[str]:
    """The XPointer, in the file a KOReader device holds, of a Kobo span.

    ``kepub_path`` is the KEPUB the Kobo holds and ``source``/``span_id`` its
    ``Location``; the span reaches ``epub_path`` through the text the two
    books share (``kepub_alignment``), and only if ``device_digest`` is the
    digest of ``epub_path``.
    """
    if not isinstance(device_digest, str) or not device_digest:
        return None
    from .kepub_alignment import span_to_xpointer
    try:
        path, kepub = os.fspath(epub_path), os.fspath(kepub_path)
        before = (_identity(path), _identity(kepub))
        if file_digest(path) != device_digest.lower():
            return None
        xpointer = span_to_xpointer(path, kepub, source, span_id)
        if xpointer is None or (_identity(path), _identity(kepub)) != before:
            return None
    except (OSError, TypeError):
        return None
    return xpointer


def kobo_position_for_device(*, user_id, book_id, record, document) -> Optional[str]:
    """The XPointer, in the requesting device's file, of a Kobo's place.

    ``record`` is the ``kosync_progress`` row about to be served. It
    qualifies only when a Kobo wrote it (a percentage-only row) and the span
    behind it is known for certain:

    * the user's Kobo bookmark is a ``KoboSpan`` at exactly the row's
      percentage, and it is the latest report of one of the user's Kobos
      (that device's journal row holds the same location and percentage), so
      the span is the place the row was made from, not a place the server
      re-placed since (a re-anchored book arms a latch and moves the bookmark
      away from the device's own report);
    * that Kobo downloaded the book as a KEPUB no earlier than the library
      KEPUB was last written, so the span ids it reports are this file's.

    The XPointer is then derived per request, and only if ``document`` -- the
    digest the device asked with -- is the library EPUB's
    (``kobo_device_xpointer``).
    """
    from .. import calibre_db, ub
    from ..annotations import _book_format_path
    from ..progress_syncing.protocols.kosync import PERCENTAGE_ONLY_LOCATOR
    from .parallel import run_blocking

    if (record is None or record.progress != PERCENTAGE_ONLY_LOCATOR
            or record.device != KOBO_DEVICE or record.percentage is None):
        return None
    bookmark = ub.session.query(ub.KoboBookmark).join(
        ub.KoboReadingState,
        ub.KoboReadingState.id == ub.KoboBookmark.kobo_reading_state_id,
    ).filter(
        ub.KoboReadingState.user_id == int(user_id),
        ub.KoboReadingState.book_id == int(book_id),
    ).first()
    if (bookmark is None or bookmark.location_type != "KoboSpan"
            or not bookmark.location_source or not bookmark.location_value
            or bookmark.progress_percent != record.percentage):
        return None
    reporters = [device_id for (device_id,) in ub.session.query(
        ub.DeviceReadingPosition.device_id,
    ).join(
        ub.Device, ub.Device.id == ub.DeviceReadingPosition.device_id,
    ).filter(
        ub.Device.user_id == int(user_id),
        ub.Device.kind == "kobo",
        ub.DeviceReadingPosition.book_id == int(book_id),
        ub.DeviceReadingPosition.location_type == bookmark.location_type,
        ub.DeviceReadingPosition.location_source == bookmark.location_source,
        ub.DeviceReadingPosition.location_value == bookmark.location_value,
        ub.DeviceReadingPosition.progress_percent == record.percentage,
    )]
    if not reporters:
        return None
    book = calibre_db.get_book(int(book_id))
    if book is None:
        return None
    epub_path = _book_format_path(book, "EPUB")
    kepub_path = _book_format_path(book, "KEPUB")
    if not epub_path or not kepub_path:
        return None
    if not _holds_library_kepub(reporters, book_id, kepub_path):
        return None
    return run_blocking(lambda: kobo_device_xpointer(
        epub_path, kepub_path, bookmark.location_source, bookmark.location_value, document))


def _holds_library_kepub(device_ids, book_id, kepub_path) -> bool:
    """True when one of ``device_ids`` downloaded the book as a KEPUB no
    earlier than ``kepub_path`` was last written, so the span ids that device
    reports, and is served, are this file's."""
    from .. import ub

    try:
        written = datetime.fromtimestamp(os.stat(kepub_path).st_mtime, timezone.utc)
    except OSError:
        return False
    downloads = ub.session.query(ub.KoboDeviceBookDownload.downloaded_at).filter(
        ub.KoboDeviceBookDownload.device_id.in_(list(device_ids)),
        ub.KoboDeviceBookDownload.book_id == int(book_id),
        func.lower(ub.KoboDeviceBookDownload.book_format) == "kepub",
    ).all()
    return any(_utc(at) >= written for (at,) in downloads if at is not None)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def journal_report(session, *, device_id, book_id, document, progress, percentage,
                   observed_at) -> None:
    """Record a KOReader progress report in the per-device journal.

    The caller commits. ``document`` is the digest the device sent, kept as
    the report's ``location_source`` so a later reader can tell which file the
    XPointer addresses.
    """
    from .device_reading_position import stage_position
    stage_position(
        device_id=device_id,
        book_id=book_id,
        progress_percent=percentage,
        location_source=document,
        # A percentage-only report (no ``progress``) names no place at all.
        location_type=(KOREADER_LOCATION_TYPE if is_xpointer(progress)
                       else None if progress is None else "koreader_page"),
        location_value=progress,
        client_modified_at=observed_at,
        session=session,
    )


def kobo_device_span(epub_path, kepub_path, xpointer: str) -> Optional[tuple[str, str]]:
    """``(chapter, span id)`` of a library-EPUB XPointer in the library KEPUB.

    The inverse of ``kobo_device_xpointer``: the place crosses through the
    text the two books share (``kepub_alignment``), and neither file may
    change while it is read.
    """
    if not is_xpointer(xpointer):
        return None
    from .kepub_alignment import xpointer_to_span
    try:
        path, kepub = os.fspath(epub_path), os.fspath(kepub_path)
        before = (_identity(path), _identity(kepub))
        span = xpointer_to_span(path, kepub, xpointer)
        if span is None or (_identity(path), _identity(kepub)) != before:
            return None
    except (OSError, TypeError):
        return None
    return span


_KEPUB_INDEX_MAX = 4
_KEPUB_INDEX = OrderedDict()
_KEPUB_INDEX_LOCK = threading.Lock()


def _kepub_index(kepub_path):
    """The ``KepubIndex`` of a KEPUB, cached per file identity."""
    from .kobo_annotation_reanchor import KepubIndex
    key = (kepub_path,) + _identity(kepub_path)
    with _KEPUB_INDEX_LOCK:
        cached = _KEPUB_INDEX.get(key)
        if cached is not None:
            _KEPUB_INDEX.move_to_end(key)
            return cached
    index = KepubIndex(kepub_path)
    with _KEPUB_INDEX_LOCK:
        _KEPUB_INDEX[key] = index
        while len(_KEPUB_INDEX) > _KEPUB_INDEX_MAX:
            _KEPUB_INDEX.popitem(last=False)
    return index


def _chapter_percent(kepub_path, chapter, span_id) -> Optional[float]:
    """How far into its chapter's text ``span_id`` starts, 0-100."""
    index = _kepub_index(kepub_path)
    found = index.chapters.get(chapter) or index.chapters.get(unquote(chapter))
    if found is None or not found.text:
        return None
    for sid, start, _end in found.spans:
        if sid == span_id:
            return round(100.0 * start / len(found.text), 2)
    return None


def _kobo_span_at(epub_path, kepub_path, xpointer, percentage):
    """``(chapter, span, exact)`` a Kobo holding ``kepub_path`` resumes at.

    The XPointer's own span when it crosses exactly, otherwise the span at
    ``percentage`` of the KEPUB's text -- the same place a percentage-only
    hand-off names everywhere else, but one the Kobo can seek to.
    """
    if xpointer is not None:
        span = kobo_device_span(epub_path, kepub_path, xpointer)
        if span is not None:
            return span[0], span[1], True
    from .kobo_position_reanchor import locate_fraction
    try:
        found = locate_fraction(_kepub_index(os.fspath(kepub_path)), percentage)
    except Exception:  # noqa: BLE001 - an unreadable KEPUB keeps the stored place
        log.debug("Cannot place %.2f%% in %s", percentage, kepub_path, exc_info=True)
        return None
    return (found[0], found[1], False) if found else None


def kobo_location_for_device(*, user_id, book, bookmark, device_id) -> Optional[dict]:
    """The ``Location`` a Kobo should resume at, when the stored one is stale.

    A KOReader or web-reader push advances the shared Kobo bookmark's
    percentage but cannot write its ``Location``: that is a ``KoboSpan`` in
    the KEPUB the Kobo holds. So the bookmark kept the span of the Kobo's own
    last report, and Nickel, which seeks to ``Location``, showed the new
    percentage while opening at the old page (#1425, #1446).

    This returns a replacement only when the bookmark's percentage is
    provably another reader's (the ``kosync_progress`` row a KOReader or
    web-reader push wrote at exactly that percentage, and no Kobo of this user
    reported the stored span at it), and ``device_id`` -- the requesting Kobo --
    holds the library KEPUB as it is now (``_holds_library_kepub``), so the span
    ids it is given are its own file's. The span is that reader's exact place
    when it crosses exactly:

    * a KOReader XPointer reported for the library EPUB's own digest;
    * the web reader's CFI behind the row (``_web_reader_cfi``);

    otherwise the span at the shared percentage. ``None`` keeps the stored
    bookmark as it is, exactly as before.

    :return: ``{"Source", "Type", "Value", "ContentSourceProgressPercent",
        "exact"}``, the chapter percentage ``None`` when it cannot be measured.
    """
    from .. import ub
    from ..annotations import _book_format_path
    from ..progress_syncing.models import KOSyncProgress
    from ..progress_syncing.protocols.kosync import (PERCENTAGE_ONLY_LOCATOR,
                                                     WEB_READER_DEVICE)
    from .parallel import run_blocking

    if bookmark is None or not device_id or book is None:
        return None
    percentage = bookmark.progress_percent
    if percentage is None or percentage <= 0:
        return None
    user_id, book_id = int(user_id), int(book.id)
    with ub.session.no_autoflush:
        record = ub.session.query(KOSyncProgress).filter(
            KOSyncProgress.user_id == user_id,
            KOSyncProgress.document == str(book_id),
        ).first()
        if (record is None or record.device == KOBO_DEVICE
                or record.percentage != percentage):
            return None
        if bookmark.location_value:
            own = ub.session.query(ub.DeviceReadingPosition.id).join(
                ub.Device, ub.Device.id == ub.DeviceReadingPosition.device_id,
            ).filter(
                ub.Device.user_id == user_id,
                ub.Device.kind == "kobo",
                ub.DeviceReadingPosition.book_id == book_id,
                ub.DeviceReadingPosition.location_type == bookmark.location_type,
                ub.DeviceReadingPosition.location_source == bookmark.location_source,
                ub.DeviceReadingPosition.location_value == bookmark.location_value,
                ub.DeviceReadingPosition.progress_percent == percentage,
            ).first()
            if own is not None:
                return None  # a Kobo reached this percentage at this span itself
        epub_path = _book_format_path(book, "EPUB")
        kepub_path = _book_format_path(book, "KEPUB")
        if not kepub_path or not _holds_library_kepub([int(device_id)], book_id, kepub_path):
            return None
        xpointer = cfi = digests = None
        if record.progress == PERCENTAGE_ONLY_LOCATOR:
            if record.device == WEB_READER_DEVICE:
                cfi = _web_reader_cfi(user_id, book_id, percentage)
        elif is_xpointer(record.progress):
            xpointer = record.progress
            digests = {digest.lower() for (digest,) in ub.session.query(
                ub.DeviceReadingPosition.location_source,
            ).join(
                ub.Device, ub.Device.id == ub.DeviceReadingPosition.device_id,
            ).filter(
                ub.Device.user_id == user_id,
                ub.Device.kind == "koreader",
                ub.DeviceReadingPosition.book_id == book_id,
                ub.DeviceReadingPosition.location_type == KOREADER_LOCATION_TYPE,
                ub.DeviceReadingPosition.location_value == xpointer,
                ub.DeviceReadingPosition.location_source.isnot(None),
            ) if digest}

    def resolve():
        place = None
        if epub_path and cfi:
            from .koreader_xpointer import cfi_to_xpointer
            try:
                place = cfi_to_xpointer(epub_path, cfi)
            except Exception:  # noqa: BLE001 - the percentage still applies
                log.debug("Cannot convert CFI for book %s", book_id, exc_info=True)
        elif epub_path and xpointer and digests and file_digest(epub_path) in digests:
            place = xpointer
        found = _kobo_span_at(epub_path, kepub_path, place, percentage)
        if found is None:
            return None
        chapter, span_id, exact = found
        return {
            "Source": chapter, "Type": "KoboSpan", "Value": span_id,
            "ContentSourceProgressPercent": _chapter_percent(kepub_path, chapter, span_id),
            "exact": exact,
        }

    return run_blocking(resolve)
