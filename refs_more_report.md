# refs_more.txt: what it holds and why

Built 2026-10-25 as the second reference list for riddim (part 1 is `refs.txt`, 51 lines). Format is the same: one URL followed by its tags. A trailing `# ...` comment holds a note, and `ml/collect_refs.py` strips it. No URL in this file also appears in `refs.txt`.

## Counts

**224 entries in total.** 210 are archive.org items. Each one was checked against `https://archive.org/metadata/<id>` and its `files` list contains audio (MP3, FLAC, OGG or AIFF). The other 14 are official label pages on Bandcamp. For each of those I checked that the page exists and that its title names the label.

| family (first tag) | entries | notes |
|---|---|---|
| dubstep | 76 | about 10 releases (DMZ, Big Apple, Hyperdub, Planet Mu, Echodub, Tectonic/Tempa/Sentry on Bandcamp). The rest are radio shows and sets from 2002–2011: Breezeblock, Essential Mix, Kiss, Rinse, FWD>>, Sub FM |
| roots_dub | 48 | Tubby/Yabby You, Black Ark, Pablo, Revolutionaries, Roots Radics, Hudson, Prince Far I, Aswad, Bovell, Creation Rebel, Greensleeves 12" run, Pressure Sounds |
| steppers_dub | 25 | Jah Shaka dance tapes from 1984–1996, Shaka vs Fatman clash, Mad Professor/Ariwa, Aba Shanti-I, Channel One at Carnival, Vibronics, Mungo's |
| grime | 23 | grimetapes.com pirate tapes (Wiley, Dizzee/Slimzee, Ruff Sqwad, Meridian, Geeneus), Rinse/Raw Blaze/Desire FM sets from 2004–2009 |
| uk_garage | 15 | Artful Dodger live and cassette mix, Todd Edwards Essential Mix, the Todd/MJ Cole/Sunship/Zed Bias show, Norris Windross tapes, Heartless Crew, Sub FM old-skool specials, Zomby bassline |
| digital_dub | 12 | Jahtari (CC), Baroque Dub (CC), Coldtear free compilations |
| dub_techno | 7 | Basic Channel *Quadrant Dub*, Rhythm & Sound, Echospace, Scion |
| speed_garage | 6 | 187 Lockdown, Scott Garcia, Lyndon Edwards classics mixes |
| roots_reggae | 6 | vocal LPs whose riddims hold up the dubs (Burning Spear, Aswad, Mighty Diamonds, LKJ/Bovell, Misty) |
| dark_garage | 5 | Oris Jay, El-B, J Da Flex, Slaughter Mob, Mac 10 |
| soundclash | 1 | King Addies vs Trevor Saxon (1994) |

The other tag axes are `radio`, `live`, `mix` and `soundsystem`, so sets can be pulled out on their own (see "Use" below).

Sources: archive.org has 210 entries and Bandcamp has 14. There are no YouTube entries; see the gaps section.

## Licences

| licence tag | entries | meaning |
|---|---|---|
| `study_only` | 199 | The item has no `licenseurl`. These are fan uploads of commercial records and radio rips, so use them for personal listening analysis and judge training only. Do not redistribute the audio or any model output that reproduces it. |
| `cc-by-nc-nd` | 12 | Jahtari, Baroque Dub, Echodub, Luscombe radio, Art of Beatz Vex'd, dub techno mix |
| `cc-by-nd` | 4 | Basic Channel *Quadrant Dub*, Burial *Street Halo*, Echospace, Sly Dunbar *Sly, Wicked and Slick*. The uploader set these labels; they are not the rights holders' own grants. |
| `cc0` | 3 | King Tubby & Yabby You, Lee Perry *Scratch in Dub*, Pinch Tectonic mix. These are also uploader claims, so treat them as study_only in practice. |
| `public_domain` | 2 | Jah Shaka at SOB's 1998, the grimetapes.com archive. Same caveat. |
| `free-dl` | 3 | Jahtari *Secret Level* (name your price on Bandcamp), Coldtear *Heavy Dub* volumes 1 and 2 (free netlabel compilations) |

Only the netlabel releases have licences that can be trusted: Jahtari, Baroque Dub, Echodub and Coldtear. Wherever a CC tag sits on a commercial record (Basic Channel, Burial, Sly Dunbar, the cc0 Tubby/Perry items), the uploader chose it, so treat that item as study-only as well. Nothing here should end up in a shipped model's training set without that caveat. The manifest already records where every file came from.

## Size warnings

Some items are much larger than an album. The collector fetches whole items, so expect these to take a while:

- `grimetapes`: about 4.1 GB, roughly 40 hours of tapes
- `greensleeves_dub_jamaican`: about 2.9 GB, 403 tracks
- `wedge-on-sub-fm-2007`: about 1.5 GB
- `threnody-on-sub-fm-2007`: about 1.5 GB
- `cve-1`: about 1.3 GB, with WAV
- `artful-dodger-garage-mix-vol.-1-cassette`: about 1.4 GB, because the original is AIFF (`--prefer-mp3` avoids this)
- `ajc03612_dubsyndicate_1996-02-10`: about 1.2 GB, 24-bit FLAC

The full list comes to roughly 40 GB without `--prefer-mp3`.

## Gaps: canonical records I could not find in a usable form

archive.org has lost most commercial reggae and UKG albums to takedowns. These were searched for and are missing, or exist only as radio or fan mixes:

- **Roots dub.** None of the Scientist LPs beyond *Rids the World...* and *Dub Duel*, which are already in refs.txt (no *Heavyweight Dub Champion*, *Meets the Space Invaders* or *Scientific Dub*). No Prince Jammy LPs (*In Lion Dub Style*, *Kamikazi Dub*). None of the Joe Gibbs/Errol Thompson *African Dub* chapters, and no *Blackboard Jungle Dub*, *Garvey's Ghost* (that item is empty), *East of the River Nile*, Glen Brown, Wackies or Blood & Fire reissues. The Algoriddim and Disciples radio specials for Niney, Joe Gibbs and Yabby You do exist; I left them out as talk-heavy.
- **UK steppers.** No studio albums by Iration Steppas, Zion Train, The Disciples, Alpha & Omega, Aba Shanti or Conscious Sounds. The live dance tapes partly make up for this. The "Alpha & Omega" items on archive.org are other acts (a DnB producer and a space-rock band), and I excluded them.
- **On-U.** No *New Age Steppers*, no African Head Charge (two items exist but contain no audio) and no Dub Syndicate studio albums (the items are mislabelled compilations, so I excluded them).
- **Dub techno.** No *BCD* compilation and no Rhythm & Sound albums (*w/ the Artists*, *See Mi Yah*). There is one Burial Mix special and one "ER" upload.
- **Dubstep releases.** No Soul Jazz *Box of Dub*, no Skull Disco or Shackleton, no Loefah 12"s, no Deep Medi catalogue, and no Pinch *Underwater Dancehall*. Tectonic, Tempa and Sentry appear only as Bandcamp pages. The "MALA" items on archive.org (Monofonicos/sern netlabels) are a different artist, and I excluded them.
- **UK garage.** There are no albums at all: nothing by Todd Edwards, MJ Cole (*Sincere*), Artful Dodger, Horsepower Productions, El-B, Zed Bias, Groove Chronicles, Tuff Jam or Dem 2, and no Armand Van Helden speed-garage 12"s. No dated Kiss FM garage shows from 1997–2001 either. The section is mostly DJ mixes and tapes.
- **Grime instrumentals.** No Wiley eskibeat 12"s (*Eskimo*, *Ice Rink*), no Ruff Sqwad, Jammer or Musical Mob instrumentals. The vocal pirate tapes contain the beats, but under MCs.
- **FWD>> 2003–05.** The earliest FWD>> sets I found are from 2009.
- **YouTube.** I added none. I could not confirm video IDs on official Boiler Room, Rinse or Deep Medi uploads without a working search, and I did not want guessed URLs in the file. If you want them, add them by hand with the `yt` tag. Good candidates: Mala at Boiler Room, Rinse FM's own channel archive, and Deep Medi's official channel.

If a future pass can buy or rip the user's own copies, the first priorities are the items above. Ranked by value to riddim: *Heavyweight Dub Champion*, *African Dub Chapter 3*, *Box of Dub*, *Rhythm & Sound w/ the Artists*, *Horsepower: In Fine Style*, MJ Cole *Sincere*, and Wiley *Eskimo*.

## Use

```
python ml/collect_refs.py refs_more.txt --out ~/refs --prefer-mp3
```

The radio and live sets are tagged `radio`, `live` or `soundsystem`. Filter on those tags to study long-form set structure (energy curve, track lengths, how long a mix overlap runs, how the selector rewinds, how the MC rides the riddim). Filter on releases to study the tracks themselves.

## 20 must-study items and what riddim should learn from each

1. **`augustus-pablo-ital-dub`**, Augustus Pablo, *Ital Dub* (1975). Tubby-desk dub at its cleanest: drum and bass carry the track while everything else drops in and out on mute switches. Use it to learn arrangement by subtraction.
2. **`the-upsetters-super-ape`**, *Super Ape* (1976). Black Ark tape-saturated low end, phaser on the whole mix, and a skank that has been pushed far back. Use it for density and haze, the opposite of Tubby's dryness.
3. **`07-yabby-u-rock-vibration`**, King Tubby / Yabby You, *Prophecy of Dub*. Spring-reverb throws on the snare, high-pass "falling" sweeps, and minor-key horn lines. Learn when an echo throw is triggered (the last offbeat of a 4- or 8-bar phrase).
4. **`keith-hudson-pick-a-dub-1994`**, *Pick a Dub* (1974). The earliest dub LP: a very slow one-drop, a bass that never stops, and sparse effects. A baseline for the minimum-viable dub.
5. **`the-revolutionaries-reaction-in-dub-1978`**, Channel One *Reaction in Dub*. Sly Dunbar's rockers and steppers drumming: four-to-the-floor kick under an offbeat skank, with the hi-hat doing the swing. This is the template for riddim's `steppers` kick grid.
6. **`roots-radics-outfit-dread-at-the-controls-dub-catalogue-1982`**, Roots Radics dub (1982). Early-80s rub-a-dub: a slower, rounder bass with fewer notes, and a single snare-delay signature. Learn how bass-line simplicity grows the sense of weight.
7. **`aswad-new-chapter-of-dub-1982`**, Aswad, *New Chapter of Dub*. UK band dub with horns sent into long delays. It shows how a melody can be cut into fragments and returned as echo.
8. **`creation-rebel-dub-from-creation-1978`**, Creation Rebel, *Dub from Creation*. Adrian Sherwood's start: harsher, more abrupt mutes and noise-like effects. Useful as the experimental edge of the `dub_fx` vocabulary.
9. **`mad-professor-dub-me-crazy_202112`**, Mad Professor, *Dub Me Crazy!!* (1982). UK digital-precision dub: clean stereo delays, tempo-synced echoes, and a glossy low end. Shows the move from tape to synced delay.
10. **`JahShakaSouthallWestLondon19thAug1988`**, a six-hour Jah Shaka dance. Soundsystem pressure as it happens in the room: siren and syndrum hits, the selector stretching one dub for minutes, and the build of a whole night. The best long-form data for live-set energy curves.
11. **`JahShaka1994DubClubTheDomeTufnellParkLondon3rdNov1994`**, Shaka at the Dome (1994). 90s UK steppers: fast steppers kicks around 140–150, digital bass and military snares. This is the direct ancestor of dubstep's sub weight.
12. **`basic-channel-quadrant-dub-01-quadrant-dub-i`**, Basic Channel, *Quadrant Dub* (1994). Dub reduced to chord-stab echoes in a filtered feedback loop. Learn the long, slow parameter automation (filter and feedback) that riddim's resampling can imitate.
13. **`DigitalMystikzdmz002JahFireVINYL`**, Digital Mystikz, *Jah Fire* (DMZ002). The template for deep dubstep: 140 half-step, a sine sub that holds the key, roots-dub samples and space. This is the reference record for riddim's `dubstep` family.
14. **`DigitalMystikz-Pathways-vinyl-2005-bossbam004`**, *Pathways* (Big Apple). Mala's sub-bass melodies, with the bass line as the lead voice under a sparse kick and snare. Learn sub-note movement on 1/8 and 1/4 steps.
15. **`Mala_Skream_Kode9_Vexd._Hatcha_Loefah_Distance.-Dubstep_Warz_Special__Breezeblock_-_10.Jan.2006`**, the *Dubstep Warz* show. Two hours of the 2006 sound, crew by crew: DMZ half-step, Tempa wobble, Vex'd distortion. Use it as a survey of the sub-styles.
16. **`Kode9__Benny_Ill-Kode9__Benny_Ill_on_GT_Radio_13-Feb-2002`**, Kode9 & Benny Ill (2002). The dark-garage-to-dubstep crossover as it happened: 2-step drums losing their shuffle and sub weight taking over. Learn where riddim's `dark_garage` and `dubstep` families meet.
17. **`Hatcha-Hatcha_studio_mix_2002`**, Hatcha studio mix (2002). Croydon dark garage played fast and loud with three-deck blends. Use it for transition length and the typical 16- or 32-bar intro and outro the DJ needs.
18. **`DizzeeDoogsMaxwellSlimzee`**, Dizzee, Doogz, Maxwell D and Slimzee (pirate tape). Early eskibeat: square-wave stabs, empty 140 grids, and the riddim built to leave room for MCs. Also a lesson in pirate pacing: 8-bar rewinds and constant reloads.
19. **`1f90y5armjo3e0otioxnbh602t2ybpvezd2qq4ks`**, Todd Edwards vs MJ Cole vs Sunship vs Zed Bias. The four 2-step producers side by side: the skipping kick (beat 1 plus the "and" of 3), shuffled hats, chopped vocals and organ stabs. This is the reference for riddim's `uk_garage` kick and shuffle.
20. **`187LockdownPresents...speedGarage`**, 187 Lockdown / Double 99 / RIP Groove speed garage. A four-to-the-floor 130 bpm house kick under a dub-reggae "Reese/donk" bass with ragga samples. Learn the `speed_garage` bass-drop structure, where the beat rolls in after the build, and how 1997 bridged dub bass and house.

Next tier, all already in the list:

- Skream & Benga at FWD>> (2009), for club-era drops and structure.
- Burial *Street Halo*, for crackle and ghost-vocal texture.
- Rhythm & Sound special Burial Mix.
- The Greensleeves 12" run, for vocal-then-dub "showcase" pairing.
- *Wiley's History of Grime*, which is narrated.
- Artful Dodger live at Glastonbury 2000, for 2-step live arrangement with an MC.
