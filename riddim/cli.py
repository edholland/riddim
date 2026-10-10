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
from .song import ALL_DELUGE, Rig
from .theory import bassline, harmony, rhythm


def _add_compose_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile", default="roots_dub",
        help="roots_dub | steppers_dub | dubstep_140 | dubby_halftime "
             "(aliases: dub, steppers, dubstep, halftime)",
    )
    parser.add_argument(
        "--key", default=None,
        help="e.g. 'G minor', 'D1 phrygian'. Omit to let the profile choose.",
    )
    parser.add_argument("--bars", type=int, default=None,
                        help="total length; sections scale proportionally")
    parser.add_argument("--seed", type=int, default=None,
                        help="same seed gives the same song")
    parser.add_argument("--progression", default=None,
                        help="force a progression by name (see `library`)")
    parser.add_argument("--template", default=None, choices=("dub", "dubstep"),
                        help="override the profile's section template")
    parser.add_argument("--all-deluge", action="store_true",
                        help="keep every part on Deluge synths instead of "
                             "sending the bass and chords out to the Coral")
    parser.add_argument("--name", default="", help="song name")
    parser.add_argument("--check", action="store_true",
                        help="run the self-critic over the song and print "
                             "what is not doing its job")
    parser.add_argument("--fix", action="store_true",
                        help="run the critic's revision loop: apply its "
                             "suggested fixes until the song is clean")


def _build(args):
    rig = ALL_DELUGE if args.all_deluge else Rig()
    return compose_song(
        profile=args.profile,
        key=args.key,
        bars=args.bars,
        seed=args.seed,
        rig=rig,
        template=args.template,
        progression=args.progression,
        name=args.name,
    )


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
        song = _apply_fix(song)
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
        song = _apply_fix(song)
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
        song = _apply_fix(song)
    print(song.summary())
    print()

    try:
        template = (
            render.parse_template(args.template_from) if args.template_from else None
        )
        rows = (
            render.parse_template(args.template_rows) if args.template_rows else None
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
                template=template, template_rows=rows,
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


def cmd_ui(args) -> int:
    from .ui.server import serve

    serve(port=args.port, open_browser=not args.no_browser)
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
        "--template-rows", default=None, metavar="ROLE=ROW,...",
        help="same as --template-from but counted as you see them on screen: "
             "row 1 is the TOP clip. The Deluge displays clips in reverse of "
             "their internal order, so this is usually what you want.",
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
        help="roots_dub | steppers_dub | dubstep_140 | dubby_halftime "
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

    p = subs.add_parser("ui", help="open the web UI to preview, tweak and regenerate")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
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
