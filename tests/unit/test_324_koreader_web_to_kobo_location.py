# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
"""A KOReader or web-reader position reaches a Kobo as a span it seeks to (#324, #1425, #1446).

A KOReader or web-reader push advances the shared Kobo bookmark's percentage
but cannot write its ``Location`` (a ``KoboSpan`` in the KEPUB the Kobo holds),
so the Kobo used to be told the new percentage with the span of its own last
report, and Nickel opened at that old page.

Driven through the real routes: KOReader's progress PUT, the web reader's
position write, and the Kobo's reading-state GET/PUT (``HandleStateRequest``),
over a real SQLite app database, with ``alice-pg11.epub`` as the library EPUB
and ``alice-pg11.kepub.epub`` as the library KEPUB the Kobo downloaded.

What must hold:
* the Kobo is served the other reader's exact span when the place provably
  crosses (KOReader on the library EPUB, or the web reader's own CFI), and the
  span at the shared percentage otherwise -- never the stale span;
* only for a Kobo that provably holds the library KEPUB as it is now, and only
  when the bookmark's percentage is provably another reader's; a Kobo's own
  place is served exactly as stored, as before;
* which position wins is unchanged.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import flask
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cps import calibre_db, config, helper, ub
from cps.progress_syncing.checksums.koreader import calculate_koreader_partial_md5
from cps.services import kobo_post_download_restore, koreader_xpointer, reading_position
from tests.unit.test_koreader_xpointer import FIXTURES

pytestmark = pytest.mark.unit

BOOK_ID = 11
BOOK_UUID = "0b2c6a2e-0000-4000-8000-000000000011"
OTHER_FILE = "0" * 32


def _span(n):
    row = json.loads((FIXTURES / "alice-pg11.kobo-spans.json").read_text(encoding="utf-8"))[n]
    return row["source"], row["span"], row["xpointer"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    kosync = importlib.import_module("cps.progress_syncing.protocols.kosync")
    kobo = importlib.import_module("cps.kobo")

    engine = create_engine("sqlite:///" + str(tmp_path / "app.db"))
    ub.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(ub, "session", session)
    monkeypatch.setattr(ub, "session_flush", lambda *a, **k: session.flush() is None)
    monkeypatch.setattr(ub, "session_commit", lambda *a, **k: session.commit() is None)
    user = ub.User(name="maggie", email="maggie@example.com", role=0, password="x")
    session.add(user)
    session.commit()
    libra = ub.Device(user_id=user.id, kind="kobo", display_name="Kobo Libra")
    clara = ub.Device(user_id=user.id, kind="kobo", display_name="Kobo Clara")
    browser = ub.Device(user_id=user.id, kind="webreader", display_name="Browser")
    session.add_all([libra, clara, browser])
    session.commit()

    library = tmp_path / "library"
    book_dir = library / "Lewis Carroll" / "Alice (11)"
    book_dir.mkdir(parents=True)
    epub = book_dir / "Alice - Lewis Carroll.epub"
    kepub = book_dir / "Alice - Lewis Carroll.kepub"
    shutil.copy(FIXTURES / "alice-pg11.epub", epub)
    shutil.copy(FIXTURES / "alice-pg11.kepub.epub", kepub)
    # The KEPUB was converted well before the Kobo downloaded it.
    written = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
    os.utime(kepub, (written, written))
    monkeypatch.setattr(config, "get_book_path", lambda: str(library))
    monkeypatch.setattr(config, "config_read_column", 0, raising=False)
    book = SimpleNamespace(
        id=BOOK_ID, uuid=BOOK_UUID, title="Alice", path="Lewis Carroll/Alice (11)",
        timestamp=datetime(2026, 1, 1),
        data=[SimpleNamespace(format="EPUB", name="Alice - Lewis Carroll"),
              SimpleNamespace(format="KEPUB", name="Alice - Lewis Carroll")])
    monkeypatch.setattr(calibre_db, "get_book", lambda _id: book)
    monkeypatch.setattr(kobo, "calibre_db", SimpleNamespace(
        get_book_by_uuid_for_kobo=lambda *a, **k: book))
    monkeypatch.setattr(kobo, "current_user", SimpleNamespace(id=user.id))
    monkeypatch.setattr(kobo, "push_reading_state_to_hardcover", lambda *a, **k: None)
    monkeypatch.setattr(helper, "queue_hardcover_reading_progress", lambda *a, **k: None)

    monkeypatch.setattr(kosync, "is_koreader_sync_enabled", lambda: True)
    monkeypatch.setattr(kosync, "authenticate_user", lambda: user)
    monkeypatch.setattr(kosync, "push_reading_state_to_hardcover", lambda *a: None)
    monkeypatch.setattr(kosync, "get_book_checksums", lambda _id: [])
    monkeypatch.setattr(kosync, "enrich_response_with_book_info",
                        lambda response, document: (response, BOOK_ID, "EPUB", book.title, "koreader"))

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(kosync.kosync)
    client = app.test_client()
    clock = [datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)]

    def kobo_download(device=libra):
        kobo_post_download_restore.record_download(
            device_id=device.id, book_id=BOOK_ID, book_format="kepub",
            log=logging.getLogger(__name__))

    def kobo_put(source, span, percent, device=libra):
        clock[0] += timedelta(minutes=5)
        payload = {"ReadingStates": [{
            "LastModified": clock[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "CurrentBookmark": {
                "ProgressPercent": percent, "ContentSourceProgressPercent": 50,
                "Location": {"Source": source, "Type": "KoboSpan", "Value": span},
            },
            "Statistics": None, "StatusInfo": None,
        }]}
        with app.test_request_context(json=payload, method="PUT"):
            flask.g.annotation_origin_device_id = device.id
            response = inspect.unwrap(kobo.HandleStateRequest)(BOOK_UUID)
        assert response.get_json()["RequestResult"] == "Success"

    def kobo_get(device=libra):
        session.expire_all()
        with app.test_request_context(method="GET"):
            flask.g.annotation_origin_device_id = device.id
            response = inspect.unwrap(kobo.HandleStateRequest)(BOOK_UUID)
        return response.get_json()[0]["CurrentBookmark"]

    def koreader_push(xpointer, percentage, document):
        response = client.put("/kosync/syncs/progress", json={
            "document": document, "progress": xpointer, "percentage": percentage,
            "device": "KindleBasic5", "device_id": "kindle-1"})
        assert response.status_code == 200, response.get_json()

    def web_read(xpointer, percentage):
        cfi = koreader_xpointer.xpointer_to_cfi(str(epub), xpointer)
        assert cfi
        row = session.query(ub.Bookmark).filter_by(user_id=user.id, book_id=BOOK_ID).first()
        if row is None:
            row = ub.Bookmark(user_id=user.id, book_id=BOOK_ID, format="epub")
            session.add(row)
        row.bookmark_key = cfi
        session.flush()
        assert reading_position.record_web_reader_progress(
            user, BOOK_ID, percentage, origin_device_id=browser.id, cfi=cfi)
        session.commit()

    yield SimpleNamespace(
        session=session, user=user, libra=libra, clara=clara, kepub=kepub,
        digest=calculate_koreader_partial_md5(str(epub)),
        kobo_download=kobo_download, kobo_put=kobo_put, kobo_get=kobo_get,
        koreader_push=koreader_push, web_read=web_read)
    session.close()
    engine.dispose()


def _location(bookmark):
    location = bookmark.get("Location") or {}
    return location.get("Source"), location.get("Value")


def test_the_libra_opens_at_the_sentence_the_kindle_stopped_on(world):
    world.kobo_download()
    world.kobo_put(*_span(200)[:2], 37.0)
    source, span, xpointer = _span(300)
    world.koreader_push(xpointer, 0.62, world.digest)

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) == (source, span)
    assert bookmark["Location"]["Type"] == "KoboSpan"
    assert 0 <= bookmark["ContentSourceProgressPercent"] <= 100


def test_the_libra_opens_at_the_web_readers_sentence(world):
    world.kobo_download()
    world.kobo_put(*_span(200)[:2], 37.0)
    source, span, xpointer = _span(300)
    world.web_read(xpointer, 62.0)

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) == (source, span)


def test_a_kindle_on_another_copy_moves_the_libra_by_percentage_not_to_the_old_page(world):
    # The XPointer addresses a file the library does not hold, so its words
    # are not the library KEPUB's: the Libra goes to the percentage instead.
    world.kobo_download()
    stale = _span(200)[:2]
    world.kobo_put(*stale, 37.0)
    exact_source, exact_span, xpointer = _span(300)
    world.koreader_push(xpointer, 0.62, OTHER_FILE)

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) not in (stale, (exact_source, exact_span))
    # A span of the library KEPUB, further on than the old one.
    sources = [r["source"] for r in json.loads(
        (FIXTURES / "alice-pg11.kobo-spans.json").read_text(encoding="utf-8"))]
    assert bookmark["Location"]["Source"] in set(sources)
    assert sources.index(bookmark["Location"]["Source"]) > sources.index(stale[0])
    assert bookmark["Location"]["Value"].startswith("kobo.")


def test_a_web_reader_with_no_known_cfi_moves_the_libra_by_percentage(world):
    kosync = importlib.import_module("cps.progress_syncing.protocols.kosync")
    world.kobo_download()
    stale = _span(200)[:2]
    world.kobo_put(*stale, 37.0)
    kosync.update_book_read_status(world.user, BOOK_ID, 62.0)
    kosync.record_percentage_only_progress(world.user.id, BOOK_ID, 62.0, device="Web reader")
    world.session.commit()

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) != stale
    assert bookmark["Location"]["Value"].startswith("kobo.")


def test_a_kobo_with_no_recorded_kepub_download_is_served_the_bookmark_as_stored(world):
    # Nothing says which file the Clara holds, so no span id can be its own.
    world.kobo_download()
    stale = _span(200)[:2]
    world.kobo_put(*stale, 37.0)
    world.koreader_push(_span(300)[2], 0.62, world.digest)

    bookmark = world.kobo_get(device=world.clara)

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) == stale


def test_a_kepub_rewritten_since_the_download_is_served_the_bookmark_as_stored(world):
    world.kobo_download()
    stale = _span(200)[:2]
    world.kobo_put(*stale, 37.0)
    world.koreader_push(_span(300)[2], 0.62, world.digest)
    later = (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp()
    os.utime(world.kepub, (later, later))  # re-converted after the download

    assert _location(world.kobo_get()) == stale


def test_the_kobos_own_place_is_served_as_stored(world):
    world.kobo_download()
    world.koreader_push(_span(100)[2], 0.2, world.digest)
    own = _span(200)[:2]
    world.kobo_put(*own, 37.0)  # ahead of the Kindle: the Kobo's own place

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 37
    assert _location(bookmark) == own
    assert bookmark["ContentSourceProgressPercent"] == 50


def test_a_kindle_behind_the_kobo_does_not_move_it(world):
    world.kobo_download()
    own = _span(300)[:2]
    world.kobo_put(*own, 62.0)
    world.koreader_push(_span(200)[2], 0.37, world.digest)  # behind: furthest wins

    bookmark = world.kobo_get()

    assert bookmark["ProgressPercent"] == 62
    assert _location(bookmark) == own


def test_the_kobos_report_from_the_served_place_takes_over(world):
    world.kobo_download()
    world.kobo_put(*_span(200)[:2], 37.0)
    source, span, xpointer = _span(300)
    world.koreader_push(xpointer, 0.62, world.digest)
    assert _location(world.kobo_get()) == (source, span)

    later = _span(320)[:2]
    world.kobo_put(*later, 65.0)  # the Libra read on from where it was put

    bookmark = world.kobo_get()
    assert bookmark["ProgressPercent"] == 65
    assert _location(bookmark) == later
