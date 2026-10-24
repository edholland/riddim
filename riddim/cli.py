"""Command line front end.

    python -m riddim compose --profile dubstep --key "G phrygian" --out sketch.mid
    python -m riddim plan --profile dub --seed 42
    python -m riddim send --profile dub --seed 42
    python -m riddim library
"""

from __future__ import annotations

import argparse
import sys

from . import profile as profile_mod
from .compose import compose as compose_song
from .deluge.render import DEFAULT_TEMPLATE_ROWS
from .song import ALL_DELUGE, Rig
from .theory import bassline, harmony, rhythm


def _add_compose_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile", default="roots_dub",
        help="roots_dub | steppers_dub | dubstep_140 | dubby_halftime | uk_garage | dark_garage "
             "(aliases: dub, steppers, dubstep, halftime)",
    )
    parser.add_argument(
        "--key", default=None,
        help="e.g. 'G minor', 'D1 phrygian'. Omit to let the profile choose.",
    )
    parser.add_argument("--bars", type=int, default=None,
                        help="total length; sections scale proportionally")
    parser.add_argument("--bpm", type=int, default=None,
                        help="tempo; omit to draw one from the profile's range")
    parser.add_argument("--seed", type=int, default=None,
                        help="same seed gives the same song")
    parser.add_argument("--progression", default=None,
                        help="force a progression by name (see `library`)")
    parser.add_argument("--template", default=None, choices=("dub", "dubstep"),
                        help="override the profile's section template")
    parser.add_argument("--all-deluge", action="store_true",
                        help="keep every part on Deluge synths instead of "
                             "sending the bass and chords out to the Coral")
    parser.add_argument("--live", action="store_true",
                        help="a live track: short loopable sections, straight in at "
                             "energy, the break only where you launch it")
    parser.add_argument("--name", default="", help="song name")
    parser.add_argument("--check", action="store_true",
                        help="run the self-critic over the song and print "
                             "what is not doing its job")
    parser.add_argument("--fix", action="store_true",
                        help="run the critic's revision loop: apply its "
                             "suggested fixes until the song is clean")
    _add_model_lead_args(parser)
    parser.add_argument("--params", default=None, metavar="FILE.json",
                        help="rebuild a song exactly from its saved params "
                             "(the .json a corpus writes); other song options "
                             "are ignored")


def _add_model_lead_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-lead", nargs="?", const="rhythm", default=None,
                        choices=("rhythm", "free"), metavar="STYLE",
                        help="the lead in the drops written by the trained model (see ml/): "
                             "'rhythm' (default) - its pitches on riddim's rhythm; 'free' - "
                             "its rhythm too")
    parser.add_argument("--take", type=int, default=0,
                        help="with --model-lead: which of the model's takes")


def _model_lead(args, song):
    """Hand the drops' lead to the model, if asked."""
    style = getattr(args, "model_lead", None)
    if not style:
        return song
    from . import model_lead

    why = model_lead.unavailable()
    if why:
        raise SystemExit(f"error: --model-lead: {why}")
    return model_lead.write(song, take=args.take, style=style,
                            progress=lambda m: print(m, file=sys.stderr))


def _build(args):
    rig = ALL_DELUGE if args.all_deluge else Rig()
    if getattr(args, "params", None):
        from . import corpus
        song = corpus.load(args.params, rig=rig)
        if getattr(args, "live", False):
            song = compose_song(params=song.params, rig=rig, live=True)
        base = args.params.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        song.name = args.name or base
        return song if getattr(args, "fix", False) else _model_lead(args, song)
    song = compose_song(
        profile=args.profile,
        key=args.key,
        bars=args.bars,
        seed=args.seed,
        rig=rig,
        template=args.template,
        progression=args.progression,
        name=args.name,
        bpm=getattr(args, "bpm", None),
        live=getattr(args, "live", False),
    )
    # --fix recomposes from params, so the model writes after it (cmd_compose)
    return song if getattr(args, "fix", False) else _model_lead(args, song)


def _print_check(song) -> None:
    from . import critique

    findings = critique.critique(song)
    print(f"\ncheck: {critique.score(song):.2f}")
    if not findings:
        print("  clean")
        return
    for f in findings:
        print(f"  [{f.severity:<7}] {f.where:<8} {f.message}")
        for fx in f.fixes:
            note = f"  ({fx.why})" if fx.why else ""
            print(f"              -> try: {fx.describe()}{note}")


def _apply_fix(song):
    """The revision loop: worst finding, apply its fix, recompose, re-critique
    - until the song is clean or a step stops improving the score. The song
    keeps its seed throughout; only its own flaws move."""
    from . import critique

    song, steps = critique.revise(song)
    if not steps:
        print("\nfix: already clean")
        return song
    plural = "s" if len(steps) != 1 else ""
    print(f"\nfix: {len(steps)} step{plural}")
    for i, st in enumerate(steps, 1):
        kept = "" if st.applied else "   (rejected - score did not improve)"
        print(f"  {i}. {st.finding.where}: {st.finding.message}")
        print(f"     -> {st.fix.describe()}   {st.score_before:.2f} -> {st.score_after:.2f}{kept}")
    return song


def _print_ranked(ranked: list) -> None:
    from . import critique

    for rank, (song, sc) in enumerate(ranked, 1):
        fs = critique.critique(song)
        first = f"{fs[0].where}: {fs[0].message}" if fs else "clean"
        key = str(song.scale) if song.scale is not None else "?"
        prog = song.harmony.progression.name if song.harmony is not None else "?"
        print(
            f"  {rank}  seed {song.seed:<5} {key:<15} {prog:<22} "
            f"{len(song.arrangement.sections)} sections {song.bars:>3} bars  "
            f"score {sc:.2f}  [{first[:44]}]"
        )


def cmd_compose(args) -> int:
    song = _build(args)
    if args.fix:
        song = _model_lead(args, _apply_fix(song))
    print(song.summary())
    print()
    print(song.arrangement.describe(song.bpm))
    print()
    print("Performance notes:")
    for note in song.notes_for_player:
        print(f"  - {note}")

    if args.check:
        _print_check(song)
        print()

    if args.out:
        from .render import midifile

        path = midifile.write(song, args.out)
        print(f"\nwrote {path}")
    return 0


def cmd_plan(args) -> int:
    from .deluge import render

    song = _build(args)
    if args.fix:
        song = _model_lead(args, _apply_fix(song))
    print(song.summary())
    print()
    print(song.arrangement.describe(song.bpm))
    if args.check:
        _print_check(song)
        print()
    print("\nDeluge session layout:")
    print(render.plan(song))
    return 0


def cmd_send(args) -> int:
    from .deluge import render
    from .deluge.client import Deluge, DelugeError

    song = _build(args)
    if args.fix:
        song = _model_lead(args, _apply_fix(song))
    if args.no_fx_rows:
        # A kit without crash and riser rows: leave those notes behind.
        from .compose import KIT_ROWS
        fx = {KIT_ROWS["crash"], KIT_ROWS["riser"]}
        for part in song.parts:
            if part.output == "kit":
                part.notes = [n for n in part.notes if n.midi not in fx]
    print(song.summary())
    print()

    try:
        template = (
            render.parse_template(args.template_from) if args.template_from else None
        )
        rows = (
            render.parse_template(args.template_rows)
            if args.template_rows and not args.template_from else None
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        with Deluge(
            output=args.port, input=args.input, verbose=args.verbose,
            dry_run=args.dry_run,
        ) as deluge:
            if not args.dry_run and not deluge.ping():
                print(
                    "Deluge did not answer the ping. Check that the custom "
                    "firmware is running and the USB cable is a data cable.",
                    file=sys.stderr,
                )
                return 1
            report = render.send(
                song, deluge, start_index=args.start_index, mode=args.mode,
                template=template, template_rows=rows, kits=args.kits,
            )
    except DelugeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(report)
    print()
    for note in song.notes_for_player:
        print(f"  - {note}")
    return 1 if report.failures else 0


def cmd_set(args) -> int:
    """Put a whole set onto the Deluge, unattended: for each track, load the
    template song, write the track into it, save it under the track's name.
    Needs the custom firmware's LOAD_SONG and SAVE_SONG commands (19, 20)."""
    import glob
    import os
    from . import corpus
    from .deluge import render, sysex
    from .deluge.client import Deluge, DelugeError
    from .deluge.sysex import Status

    rig = ALL_DELUGE if args.all_deluge else Rig()
    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
        if not files:
            print(f"error: no .json songs in {args.dir}", file=sys.stderr)
            return 1
        tracks = []
        for path in files:
            song = corpus.load(path, rig=rig)
            if args.live:
                song = compose_song(params=song.params, rig=rig, live=True)
            title = os.path.basename(path).rsplit(".", 1)[0].replace("_", " ")
            tracks.append((title, song))
    else:
        profiles = [profile_mod.get(p).name for p in args.profiles.split(",")]
        print(f"composing {args.count} live tracks across {', '.join(profiles)}")
        entries = corpus.build(args.count, profiles, candidates=args.candidates, seed=args.seed,
                               rig=rig, progress=print, live=True)
        tracks = [(f"{i:02d} {e.title}", e.song) for i, e in enumerate(entries, 1)]
    if args.model_lead:
        tracks = [(t, _model_lead(args, song)) for t, song in tracks]
    try:
        rows = render.parse_template(args.template_rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    names = [sysex.song_name(f"{args.prefix}{t}") for t, _ in tracks]
    print(f"\n{len(tracks)} tracks -> SONGS/ on the Deluge, each from template "
          f"SONGS/{sysex.song_name(args.template)}.XML:")
    for n in names:
        print(f"  {n}")
    failures = 0
    try:
        with Deluge(output=args.port, input=args.input, verbose=args.verbose,
                    dry_run=args.dry_run) as deluge:
            if not args.dry_run and not deluge.ping():
                print("Deluge did not answer the ping.", file=sys.stderr)
                return 1
            if not args.dry_run and deluge.supports_save_load() is False:
                print("error: the Deluge is running firmware without the save/load "
                      "commands - flash build/Release/deluge.bin from the firmware "
                      "fork's ai-save-load branch (./dbt loadfw).", file=sys.stderr)
                return 1
            for (title, song), name in zip(tracks, names):
                print(f"\n== {name}")
                r = deluge.load_song(args.template)
                if r is not None and not r.ok:
                    why = {Status.BUSY: "stop playback and leave any menu, then run again",
                           Status.FILE_NOT_FOUND: f"no SONGS/{args.template}.XML on the card"}
                    print(f"   load template failed: {r.status.name} - {why.get(r.status, '')}")
                    return 1
                report = render.send(song, deluge, mode=args.mode, template_rows=rows,
                                     progress=False, kits=args.kits)
                print(f"   sent: {report.clips} clips, {report.notes} notes"
                      + (f", {len(report.failures)} failures"
                         if report.failures and not args.dry_run else ""))
                armed = deluge.arm_for_save(song)
                if armed is not None and not armed.ok:
                    print(f"   arm first section: {armed.status.name} (the song may start on "
                          f"the template's empty clips)")
                r = deluge.save_song(name, overwrite=args.overwrite)
                if r is not None and not r.ok:
                    failures += 1
                    hint = {Status.FILE_EXISTS: "already on the card (use --overwrite)",
                            Status.NEEDS_UI_SAVE: "the template uses collected/recorded samples - "
                                                  "save it once from the Deluge's Save menu"}
                    print(f"   save failed: {r.status.name} {hint.get(r.status, '')}")
                else:
                    print(f"   saved SONGS/{name}.XML")
    except DelugeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\n{len(tracks) - failures}/{len(tracks)} tracks saved.")
    return 1 if failures else 0


def cmd_ping(args) -> int:
    from .deluge.client import Deluge, DelugeError

    try:
        with Deluge(output=args.port, input=args.input, verbose=True) as deluge:
            ok = deluge.ping()
    except DelugeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("Deluge responded." if ok else "No response.")
    return 0 if ok else 1


def cmd_ports(args) -> int:
    import mido

    print("outputs:")
    for name in mido.get_output_names():
        print(f"  {name}")
    print("inputs:")
    for name in mido.get_input_names():
        print(f"  {name}")
    return 0


def cmd_batch(args) -> int:
    """Compose a batch, let the critic score it, and rank.

    Selection pressure with the engine's own knowledge: same profile, a run
    of consecutive seeds (deterministic), each scored by how well it does
    its job. No model, no dataset - just the rules voting."""
    from . import critique

    ranked = []
    for seed in range(args.seed, args.seed + args.count):
        args.seed = seed
        ranked.append((_build(args), 0.0))
    for i, (song, _) in enumerate(ranked):
        ranked[i] = (song, critique.score(song))
    ranked.sort(key=lambda t: (-t[1], t[0].seed))

    print(f"{args.profile}: {args.count} songs, scored by the critic")
    top = ranked[: max(1, args.top)]
    _print_ranked(top)

    if args.fix:
        plural = "s" if len(top) != 1 else ""
        print(f"\nrevising the top {len(top)}")
        fixed = []
        for song, sc in top:
            song, steps = critique.revise(song)
            fixed.append((song, critique.score(song)))
            print(f"  seed {song.seed}: {sc:.2f} -> {fixed[-1][1]:.2f}  "
                  f"({len(steps)} step{'s' if len(steps) != 1 else ''})")
        fixed.sort(key=lambda t: (-t[1], t[0].seed))
        print("ranked after fixing:")
        _print_ranked(fixed)
        top = fixed

    if args.out:
        from .render import midifile

        path = midifile.write(top[0][0], args.out)
        print(f"\nwrote {path}")
    return 0


def cmd_corpus(args) -> int:
    """Compose a show: many songs, best of several seeds each, ordered as a
    set, written as MIDI + params + plan, with a set list."""
    from . import corpus

    profiles = [profile_mod.get(p).name for p in args.profiles.split(",")]
    rig = ALL_DELUGE if args.all_deluge else Rig()
    print(f"composing {args.count} songs across {', '.join(profiles)} "
          f"(best of {args.candidates} each)")
    entries = corpus.build(args.count, profiles, candidates=args.candidates,
                           seed=args.seed, fix=not args.no_fix, rig=rig, progress=print,
                           live=args.live)
    path = corpus.write(entries, args.out, audio=args.audio, progress=print)
    print()
    print(open(path).read())
    print(f"wrote {len(entries)} songs to {args.out}/  (set list: {path})")
    return 0


def cmd_vary(args) -> int:
    """Variations on a song you like: mutate its knobs, or re-roll one of its
    seeds (hook, groove, structure...), keep the rest. Ranked, written out."""
    from . import corpus

    rig = ALL_DELUGE if args.all_deluge else Rig()
    base = _build(args)
    print(f"varying {base.name}: {args.count} x {args.how}")
    entries = corpus.variations(base, args.count, args.how, seed=args.vary_seed, rig=rig)
    for e in entries:
        print(f"  {e.title:<40} score {e.score:.3f}")
    path = corpus.write(entries, args.out, audio=args.audio, progress=print)
    print(f"wrote {len(entries)} variations to {args.out}/  (list: {path})")
    return 0


def cmd_deluge_kits(args) -> int:
    """The preview's drums as Deluge kits: one-shots and kit presets."""
    try:
        from .deluge import kit
    except ImportError as exc:
        print(f"deluge-kits needs the audio extra (pip install -e '.[audio]'): {exc}",
              file=sys.stderr)
        return 1
    families = tuple(args.family) if args.family else tuple(kit.KITS)
    paths = kit.export(args.out, families, seed=args.seed)
    print(f"wrote {len(paths)} files under {args.out}. Load a kit on a kit clip "
          f"(or use send --kits with firmware ai-kit-automation).")
    return 0


def cmd_render(args) -> int:
    """Render a song (or a saved params file) to a WAV for auditioning."""
    from .render import audio

    song = _build(args)
    out = args.out or f"{song.name}.wav"
    audio.write(song, out)
    print(f"wrote {out}  ({song.arrangement.duration_seconds(song.bpm) / 60:.2f} min)")
    return 0


def cmd_tells(args) -> int:
    """Where would a producer hear a machine? The tells of a batch of songs'
    first drops against each family's ranges - one profile or all of them."""
    from . import tells
    from .compose import compose

    profiles = ([profile_mod.get(args.profile).name] if args.profile else
                ["roots_dub", "steppers_dub", "dubby_halftime", "uk_garage",
                 "dark_garage", "dubstep_140"])
    for prof in profiles:
        songs = [compose(profile=prof, seed=s, live=args.live)
                 for s in range(args.seed, args.seed + args.count)]
        print(f"{prof}: {len(songs)} songs, "
              f"{sum(tells.count(s) for s in songs) / len(songs):.2f} tells a song")
        for name, row in tells.survey(songs).items():
            print(f"  {name:14s} mean {row['mean']:6.2f}   human-range {row['pass']:4.0%}")
    return 0


def cmd_ui(args) -> int:
    from .ui.server import serve

    serve(port=args.port, open_browser=not args.no_browser, live=args.live)
    return 0


def cmd_library(args) -> int:
    """Print the theory library. This doubles as the documentation."""
    print("PROFILES")
    for name, prof in profile_mod.PROFILES.items():
        print(f"  {name:<16} {prof.bpm:>3}bpm  {prof.drum_style:<16} "
              f"sub {prof.sub}")
        if prof.note:
            print(f"      {prof.note}")

    print("\nPROGRESSIONS  (degrees are 0-indexed scale steps)")
    for prog in harmony.PROGRESSIONS:
        degrees = "-".join(str(d) for d in prog.degrees)
        print(f"  {prog.name:<20} {degrees:<12} tension {prog.tension:.2f}  "
              f"[{', '.join(prog.modes)}]")
        if prog.note:
            print(f"      {prog.note}")

    print("\nBASS FIGURES")
    for fig in bassline.FIGURES:
        print(f"  {fig.name:<18} {len(fig.notes)} notes")
        if fig.note:
            print(f"      {fig.note}")

    print("\nDRUM RIDDIMS")
    for name, voices in rhythm.DRUM_RIDDIMS.items():
        print(f"  {name}")
        for voice, pattern in voices.items():
            print(f"      {voice:<6} {pattern}")

    print("\nCLAVES")
    for name, pattern in rhythm.CLAVES.items():
        print(f"  {name:<12} {pattern}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riddim",
        description="Theory-grounded dub and dubstep composition for the "
                    "Deluge and a modular rack.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("compose", help="generate a song and write a MIDI file")
    _add_compose_args(p)
    p.add_argument("--out", default=None, help="write a .mid file here")
    p.set_defaults(func=cmd_compose)

    p = subs.add_parser("plan", help="show what would be sent, without sending")
    _add_compose_args(p)
    p.set_defaults(func=cmd_plan)

    p = subs.add_parser("send", help="generate and write into the Deluge session")
    _add_compose_args(p)
    p.add_argument("--port", default=None, help="MIDI output port name or substring")
    p.add_argument("--input", default=None, help="MIDI input port for replies")
    p.add_argument("--start-index", type=int, default=None,
                   help="first session clip index to write to "
                        "(default: append after the existing clips)")
    p.add_argument(
        "--template-rows", default=DEFAULT_TEMPLATE_ROWS, metavar="ROLE=ROW,...",
        help="same as --template-from but counted as you see them on screen: "
             "row 1 is the TOP clip. The Deluge displays clips in reverse of "
             f"their internal order, so this is usually what you want. Default "
             f"{DEFAULT_TEMPLATE_ROWS}; pass '' to create new instruments instead.",
    )
    p.add_argument(
        "--template-from", default=None, metavar="ROLE=INDEX,...",
        help="clone each role from an existing clip instead of creating a new "
             "instrument, e.g. 'chords=0,sub=1,lead=2,pad=3'. The generated "
             "clips inherit that clip's instrument - MIDI channel, patch and "
             "sends - which is the only way to control a MIDI clip's output "
             "channel.",
    )
    p.add_argument(
        "--mode", default="jam", choices=("jam", "song", "none"),
        help="jam: each section loops until you launch the next (default). "
             "song: sections auto-advance and the arrangement plays itself. "
             "none: leave section repeats alone.",
    )
    p.add_argument("--no-fx-rows", action="store_true",
                   help="drop the crash (row 42) and riser (row 43) notes, for "
                        "a drum kit that has no sounds on those rows")
    p.add_argument("--kits", action="store_true",
                   help="load the family's riddim kit (KITS/RIDDIM DUB.XML etc., from "
                        "`deluge-kits`) on the drum and perc clips and write the kit's "
                        "desk moves - sweeps, echo throws (firmware ai-kit-automation)")
    p.add_argument("--dry-run", action="store_true",
                   help="run the whole send path without opening a port")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_send)

    p = subs.add_parser("ping", help="check the Deluge is listening")
    p.add_argument("--port", default=None)
    p.add_argument("--input", default=None)
    p.set_defaults(func=cmd_ping)

    p = subs.add_parser("ports", help="list MIDI ports")
    p.set_defaults(func=cmd_ports)

    p = subs.add_parser(
        "batch", help="compose a batch of songs, ranked by the critic's score")
    p.add_argument(
        "--profile", default="roots_dub",
        help="roots_dub | steppers_dub | dubstep_140 | dubby_halftime | uk_garage | dark_garage "
             "(aliases: dub, steppers, dubstep, halftime)",
    )
    p.add_argument("--key", default=None,
                   help="e.g. 'G minor', 'D1 phrygian'. Omit to let the profile choose.")
    p.add_argument("--seed", type=int, default=1,
                   help="first seed; the batch runs consecutive seeds from it")
    p.add_argument("--count", type=int, default=20,
                   help="how many to compose and score")
    p.add_argument("--top", type=int, default=5,
                   help="how many to show (and to fix with --fix)")
    p.add_argument("--bars", type=int, default=None,
                   help="total length; sections scale proportionally")
    p.add_argument("--bpm", type=int, default=None,
                   help="tempo; omit to draw one per song from the profile's range")
    p.add_argument("--progression", default=None,
                   help="force a progression by name (see `library`)")
    p.add_argument("--template", default=None, choices=("dub", "dubstep"),
                   help="override the profile's section template")
    p.add_argument("--all-deluge", action="store_true",
                   help="keep every part on Deluge synths instead of "
                        "sending the bass and chords out to the Coral")
    p.add_argument("--name", default="", help="song name")
    p.add_argument("--fix", action="store_true",
                   help="run the revision loop over the top songs")
    p.add_argument("--out", default=None,
                   help="write the best song's MIDI here")
    p.set_defaults(func=cmd_batch)

    p = subs.add_parser(
        "set", help="put a whole set onto the Deluge unattended: load template, "
                    "write track, save - per track (custom firmware 19/20)")
    p.add_argument("--dir", default=None,
                   help="a corpus folder of .json songs (in file order); omit to "
                        "compose a fresh live set")
    p.add_argument("--live", action="store_true",
                   help="with --dir: turn each song into a live track")
    p.add_argument("--count", type=int, default=12, help="tracks to compose (no --dir)")
    p.add_argument("--profiles", default=",".join(
        ("roots_dub", "steppers_dub", "dubby_halftime", "uk_garage",
         "dark_garage", "dubstep_140")), help="profiles to spread a composed set across")
    p.add_argument("--candidates", type=int, default=6, help="seeds auditioned per track")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--template", default="RIDDIM TEMPLATE",
                   help="song on the card holding your instruments (SONGS/<name>.XML)")
    p.add_argument("--template-rows", default=DEFAULT_TEMPLATE_ROWS,
                   help="which row of the template holds each role's clip")
    p.add_argument("--prefix", default="", help="put before every song name, e.g. 'SET1 '")
    _add_model_lead_args(p)
    p.add_argument("--overwrite", action="store_true", help="replace songs of the same name")
    p.add_argument("--mode", default="jam", choices=("jam", "song"))
    p.add_argument("--all-deluge", action="store_true")
    p.add_argument("--port", default=None)
    p.add_argument("--input", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--kits", action="store_true",
                   help="load the family's riddim kit (KITS/RIDDIM DUB.XML etc., from "
                        "`deluge-kits`) on the drum and perc clips and write the kit's "
                        "desk moves - sweeps, echo throws (firmware ai-kit-automation)")
    p.add_argument("--dry-run", action="store_true",
                   help="compose and walk the set without a Deluge")
    p.set_defaults(func=cmd_set)

    p = subs.add_parser(
        "deluge-kits", help="write the riddim kits (samples + KITS/*.XML) for the Deluge's card")
    p.add_argument("out", help="the SD card's top level (e.g. /Volumes/DELUGE), or a folder "
                               "to copy onto it")
    p.add_argument("--family", action="append", choices=("dub", "dubstep", "garage"),
                   help="only this family's kit (repeatable; default all three)")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_deluge_kits)

    p = subs.add_parser(
        "corpus", help="compose a show: many songs, ranked, ordered as a set")
    p.add_argument("--live", action="store_true",
                   help="live tracks: short loops, straight in at energy")
    p.add_argument("--count", type=int, default=24, help="how many songs")
    p.add_argument("--profiles", default=",".join(
        ("roots_dub", "steppers_dub", "dubby_halftime", "uk_garage",
         "dark_garage", "dubstep_140")),
                   help="comma-separated profiles to spread the set across")
    p.add_argument("--candidates", type=int, default=6,
                   help="seeds auditioned per slot; the best is kept")
    p.add_argument("--seed", type=int, default=1, help="first seed")
    p.add_argument("--out", default="show", help="output directory")
    p.add_argument("--audio", action="store_true",
                   help="also render a WAV preview of every song (slower)")
    p.add_argument("--no-fix", action="store_true",
                   help="skip the critic's revision loop")
    p.add_argument("--all-deluge", action="store_true",
                   help="keep every part on Deluge synths")
    p.set_defaults(func=cmd_corpus)

    p = subs.add_parser(
        "vary", help="variations on a song: mutate it, or re-roll its hook, groove...")
    _add_compose_args(p)
    p.add_argument("--how", default="mutate",
                   help="mutate | hook | groove | arrangement | structure | drums "
                        "| bass | wild")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--vary-seed", type=int, default=1,
                   help="change this for a different batch of variations")
    p.add_argument("--out", default="variations", help="output directory")
    p.add_argument("--audio", action="store_true", help="also write audio previews")
    p.set_defaults(func=cmd_vary)

    p = subs.add_parser("render", help="render a song to a WAV preview")
    _add_compose_args(p)
    p.add_argument("--out", default=None, help="write the .wav here")
    p.set_defaults(func=cmd_render)

    p = subs.add_parser(
        "tells", help="audit a batch for the giveaways that a computer wrote it")
    p.add_argument("--profile", default=None, help="one profile (default: all six)")
    p.add_argument("--count", type=int, default=16, help="songs per profile")
    p.add_argument("--seed", type=int, default=100, help="first seed")
    p.add_argument("--live", action="store_true", help="audit live tracks")
    p.set_defaults(func=cmd_tells)

    p = subs.add_parser("ui", help="open the web UI to preview, tweak and regenerate")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--live", action="store_true",
                   help="every song is a live track: short loops, straight in at energy")
    p.set_defaults(func=cmd_ui)

    p = subs.add_parser("library", help="print the theory library")
    p.set_defaults(func=cmd_library)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
