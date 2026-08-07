# Lakoff Master Metaphor List — Vocabulary Extraction Reports

Six parallel subagents extracted canonical Lakoff image-schema vocabulary from `METAPHORLIST.pdf` (Lakoff, Espenson, & Schwartz 1991, *Master Metaphor List*, 2nd ed., Cognitive Linguistics Group, UC Berkeley). Each agent focused on one schema cluster, returned a markdown report with metaphor names, page citations, and structured Python anchor-pair lists.

The structured lists were compiled into `lakoff_canonical_vocabulary.py`. The fuller reports (this file) preserve the methodological notes, example sentences, and metaphor catalog for each cluster — material that doesn't fit cleanly into the Python module but is needed for writing up findings and for future agents who may want richer context.

Source PDF: `/Users/macn/Documents/embeddingexp/METAPHORLIST.pdf`

---

## Report 1: LIGHT-DARK

### Canonical Lakoff Metaphors Involving Light/Dark

Page references are PDF page numbers (printed page in parentheses where it differs).

1. **LIGHT IS A FLUID** — p. 179 (printed 176). Source: fluid/liquid/water. Target: light, lightness, sunlight. Ex: "Sunlight poured into the room"; "The stage was flooded with light"; "Pools of light were scattered across the clearing."

2. **DARKNESS IS A SOLID** — p. 179. Source: solid, substance. Target: dark, darkness, black, blackness. Ex: "The darkness was palpable"; "The darkness was inpenetrable"; "Darkness pressed in on all sides."

3. **DARKNESS IS A COVER** — p. 180 (printed 177). Source: cover. Target: dark, darkness. Ex: "Under cover of darkness"; "He was enveloped in darkness."

4. **LIGHT IS A LINE** — p. 180. Source: line. Target: light. Ex: "Sunbeams"; "rays of light."

5. **LIGHT MOVES FROM LIGHT SOURCE** (special case) — p. 180. Ex: "The lamp throws a lot of light"; "Where is that light coming from?"

6. **IDEAS ARE LIGHT SOURCES** — p. 100 (printed 97). Target: ideas. Ex: "What a bright idea!"; "That idea really illuminates the problem."

7. **UNDERSTANDING IS SEEING** (within IDEAS ARE PERCEPTIONS) — pp. 86-87 (printed 83-84). Subcases include:
   - **3a. Aids to Gaining Awareness are Aids to Vision**: "Can you shed more light on this issue?"; "He spotlighted the issues that were important."
   - **3c. Impediments to Awareness are Impediments to Seeing**: "I was in the dark for a long time"; "I'm just in a fog today"; "She walks around blindfolded."
   - **4c. Deception Through impairment of vision**: "They purposefully left me in the dark"; "He's deliberately clouding the issue"; "They obscured the issue"; "She blinded me with claims of innocence."
   - **6. Creativity is Seeing in a Different Light**.

8. **HOPE IS LIGHT** — p. 152 (printed 149). Source: light. Target: hope. Ex: "He has bright hopes"; "I have a very dim hope that he'll recover"; "I've been offered a ray/beam of hope"; "a glimmer of hope."

9. **STRONG EMOTION IS BLINDING** — p. 147 (printed 144). Ex: "He was blinded by love"; "My outlook was clouded with grief"; "They were dazzled with excitement."

10. **GOODNESS IS LIGHT / BADNESS IS DARKNESS** — p. 190 (printed 187). Source: light, dark, lightness, darkness, bright, brightness. Target: good, bad, goodness, badness, evil, morality. Ex: "The dark side of the force"; "The future looks brighter"; "Look on the bright side"; "He has a dark and glowering disposition."

11. **GOODNESS IS WHITE / BADNESS IS BLACK** — p. 190. Ex: "White magic/black magic"; "a black-hearted scoundrel."

12. **DARKNESS IS LIGHT'S ELDER BROTHER** (cited under A PRIOR RELATED THING IS AN OLDER SIBLING) — p. 202. Attestation: "Darkness is light's elder brother."

13. (Form file, p. 168) **A SHADOW FELL ACROSS HER BROW** — facial-expression idiom; single attestation tying *shadow* to troubled affect.

14. (Problems file, p. 197) "**The solution finally was brought to light**" — truth/discovery attestation under SOLUTION IS OBJECT BURIED IN LANDSCAPE.

### Key findings / caveats

Lakoff treats light/dark as cross-cutting many target domains — it's *not* contained in one file. The LightDark file itself is short (pp. 179-180) and focuses on the schema-level mappings (light-as-fluid, darkness-as-solid/cover, light-as-line). The rich vocabulary actually lives in **UNDERSTANDING IS SEEING** (Mental/Perception), **HOPE IS LIGHT** (Emotions/Hope), **GOODNESS IS LIGHT / BADNESS IS DARKNESS** (Other/Morality), and **IDEAS ARE LIGHT SOURCES** (Mental/Ideas). The Morality file also explicitly notes the source-domain set "light, dark, lightness, darkness, bright, brightness" mapped to "good, bad, goodness, badness, evil, morality" — strong canonical evidence for the axis. One nice attestation for the kinship asymmetry: "Darkness is light's elder brother" (p. 202).

---

## Report 2: UP-DOWN

### Canonical Lakoff UP-DOWN Metaphors Located

The MML is organized by target domain, so UP-DOWN appears scattered as a source-domain schema across many files. Confirmed canonical entries:

- **AMOUNT IS VERTICALITY / MORE IS UP** (p.14, Properties file) — "The number of people living in poverty went up." "Can you please decrease the number of assignments?" Source: verticality; Target: amount.
- **MORE IS HIGHER (= MORE IS UP)** (p.62, Comparison file) — "The number of poor people is higher than the number of rich ones." Explicitly noted as part of AMOUNT IS VERTICALITY.
- **HARMING IS LOWERING / WELL-BEING IS UP** (p.50, Harm file) — "That was a demeaning comment." "I won't do that — it's degrading!" "He debased himself." "I enjoy reading uplifting literature." Source: verticality, up; Target: harm. Subsumes HEALTHY IS UP / SICK IS DOWN in the well-being interpretation.
- **STATUS IS POSITION / HIGH STATUS IS UP** (p.59, Comparison file) — "He ranks above me." "They are lower class." "Climbing the ladder in the company." "He's the top of the heap."
- **BEING BETTER IN STATIC SITUATION IS BEING ABOVE** (p.58) — "Joe is above Bill in intelligence." "He has an edge over us." "He's up there with the best of them." Related metaphors note: "More is Up, High Status is Up."
- **CONTROL IS UP** (p.67, Competition file) — "I have control over him." "He is under my power." "I'm on top of the situation." "He has a dominating/submissive personality." (Implies OUT OF CONTROL IS DOWN.)
- **FORESEEABLE FUTURE EVENTS ARE UP** (p.76, Time) — "Upcoming events." "What's coming up this week?"
- **EUPHORIC STATES ARE UP** (p.177, Intoxication) — "He's really high." "She's coming down." Note explicitly: "This is a restricted subcase of HAPPY IS UP." This is the MML's pointer to HAPPY IS UP / SAD IS DOWN.
- **MORAL IS UP / IMMORAL IS DOWN** (p.187, Morality) — "Upright." "On the up and up." "On a high moral plain." "Above board." "High-minded." "Upstanding citizen." "Underhanded/low trick." "I wouldn't stoop to that." "That would be beneath me." "An abyss of depravity." Source: up; Target: morality.
- **GOOD IS UP / BAD IS DOWN** (p.187, Morality) — "High-quality work." "My cake will never be up there with hers." "His work has really been slipping/sliding/slumping."
- **GOODNESS IS LIGHT / BADNESS IS DARKNESS** (p.190, Morality) — included as a coherent ally of GOOD IS UP.

Also relevant supporting metaphors: CHANGE IS MOTION ("He slipped into a depression," p.15); EMOTIONAL STABILITY IS CONTACT WITH THE GROUND / IS MAINTAINING POSITION (p.148, explicitly noted as "coherent with GOOD IS UP"); ANGER IS HOT FLUID IN A CONTAINER (p.149, with "rising" entailments); CONCEIT IS INFLATION / PRIDE IS SWELLING (p.166).

### Key findings / caveats

1. The MML's organization by target domain hides the UP-DOWN schema; the richest single page is Morality (p.187) which boxes BOTH MORAL IS UP and GOOD IS UP back-to-back. Other primary deployments are scattered: Properties p.14 (MORE), Comparison pp.58-59 (STATUS, BETTER), Competition p.67 (CONTROL), Harm p.50 (WELL-BEING), Intoxication p.177 (EUPHORIC/HAPPY).
2. Several canonical primary metaphors from *Metaphors We Live By* Ch.4 — HAPPY IS UP / SAD IS DOWN, RATIONAL IS UP / EMOTIONAL IS DOWN, CONSCIOUS IS UP / UNCONSCIOUS IS DOWN, DIVINE IS UP — are **not** boxed as standalone entries in this 1991 second-draft MML; they appear only implicitly or via cross-references (e.g. the p.177 note). For embedding-axis purposes the canonical-vocabulary file uses MWLB-standard pairs for them.
3. The MML does not have a single alphabetized "INDEX" pages section despite the TOC listing one — the file appears to end at Society p.211. The TOC itself is the best entry-point.

---

## Report 3: IN-OUT / CONTAINER

### Canonical Lakoff CONTAINER / IN-OUT metaphors

**Event Structure (Location branch)**
- **STATES ARE LOCATIONS** (p 8, EventStructure/States) — "He is in love. She can stay/remain silent for days. What state is the project in?" Foundational schema: being in a state = being in a bounded region.
- **EXISTENCE IS A LOCATION (HERE)** + special case **EXISTENCE IS LOCATION OUT OF CONTAINER** (p 70, Existence) — "It came into existence. It went out of existence. The emergence of new sciences..." Existing = inside the container; ceasing = exiting it.
- **(Social Interaction is) CONTAINMENT IN A PRESCRIBED SHAPE** (p 10, States subcase of STATES ARE SHAPES) — "He doesn't fit in. She's a square peg." Group-membership-as-containment.
- **DIFFICULTIES ARE CONTAINERS** (p 75, Difficulties) — "How did I get into this situation? We're in a mess. We're in hot water. Let's get out of this situation." Canonical IN=bad valence.
- **(BOUNDED) TIME IS A CONTAINER** (p 78, Time) — "He did it in three minutes. In 1968... We're well into the century. He's like something out of the last century."

**Mental Events**
- **THE MIND IS A CONTAINER FOR OBJECTS** (p 94, MentalObj — sub-case of IDEAS ARE OBJECTS) — "I can't get this idea out of my mind. What did you have in mind?"
- **MEMORY IS A CONTAINER FOR OBJECTS** (p 95, MentalObj) — with subcases "Memorizing is Storing Objects" and "Remembering is Retrieving Objects". Note explicit "Container is a LIFO stack."
- **IDEAS ARE LOCATIONS** + **Constrained Thought is Constrained Movement** (p 90, MentalLoc) — "He's all wrapped up in his beliefs. He's tied to a belief in objectivism. That's a liberating idea."

**Emotion**
- **EMOTIONS ARE ENTITIES WITHIN A PERSON** / **BODY IS CONTAINER FOR EMOTIONS** (p 140, Emotion) — "I was filled with rage. She was overflowing with joy. She could hardly contain her anger."
- **EMOTIONS ARE LOCATIONS** (p 144, Emotion; "this is STATES ARE LOCATIONS") — licenses "in love, in despair, in a funk, in shock."
- **ANGER IS HOT FLUID IN A CONTAINER** (p 149, Anger) — "You make my blood boil. He blew his top. I can't keep my anger bottled up anymore."

**Other**
- **A PROBLEM IS A LOCKED CONTAINER FOR ITS SOLUTION** (p 195, Problems) — "The solution is contained in the problem. We have to look deeply into this problem. Discovered solution is removed from container."

That's 12 canonical Lakoff metaphors with CONTAINER/IN-OUT as source, spanning 8+ children (Mental, MentalObj, MentalLoc, States, Existence, Difficulties, Time, Emotion, Anger, Problems).

### Methodological note on valence

IN-OUT is valence-ambivalent across Lakoff children:
- **IN = good**: in love, included, admitted, retained, remembered, married, engaged, contained-emotion (composure)
- **IN = bad**: trapped in difficulty, mired, stuck, in trouble, in a mess, in despair, imprisoned
- **OUT = good** in the "bad container" cases: escaped, freed, released, extricated
- **OUT = bad** in the "good container" cases: forgotten, dismissed, expelled, exiled, divorced

A single IN-OUT axis in embedding space will fold together opposite valences, so the axis recovers an image-schematic dimension that is somewhat orthogonal to good/bad.

---

## Report 4: FORWARD-BACK / PATH / TIME-MOTION

### Canonical Lakoff metaphors with PATH / MOTION / FORWARD-BACK / JOURNEY as source

All from the EVENT STRUCTURE METAPHORICAL SYSTEM (pp. 1-79), which Lakoff frames as states-as-locations + change-as-motion + purposes-as-destinations.

| Metaphor | Page | Source → Target | Gloss / examples |
|---|---|---|---|
| **CHANGE IS MOTION** (alt: CHANGE OF STATE IS CHANGE OF LOCATION) | 4, 15 | motion / change of location → change of state | "He went from innocent to worldly"; "He slipped into a depression"; "He went back to polishing the silver" |
| **STATES ARE LOCATIONS** | 4, 8 | locations → states | "He is at a certain stage"; "She remained silent"; "He came out of the coma" |
| **PURPOSES ARE DESTINATIONS** (Desired States are Desired Locations) | 8 | destinations → purposes | "It took him hours to reach a state of perfect concentration" |
| **ACTION IS SELF-PROPELLED MOTION** | 27 | motion, moving, path → action | "He went on with what he was doing"; "She went back to sleep"; "Never looking back, he went on" |
| **PURPOSEFUL ACTION IS DIRECTED MOTION TO A DESTINATION** | 28 | directed motion → purposeful action | umbrella for the FORWARD-BACK progress metaphors |
| **PROGRESS IS FORWARD MOVEMENT / FORWARD MOTION** | 16, 28 | forward motion → progress | "Let's forge ahead"; "Let's keep moving forward"; "We fall back two steps for every one we take"; "The project is going ahead as planned" |
| **NEGATIVE PROGRESS IS BACKWARD MOVEMENT** | 29 | backward motion → regress | "I'd hate to see us back where we were thirty years ago"; "We need to backtrack"; "I keep falling behind"; "getting farther and farther from my target weight" |
| **LACK OF PROGRESS TOWARD DESTINATION IS LACK OF MOTION** | 30 | stasis → no progress | "I got stuck"; "He is at a standstill"; "Making excuses is getting me nowhere" |
| **LACK OF PURPOSE IS LACK OF DIRECTION** | 29 | aimless motion → purposelessness | "He is just floating around"; "He is drifting aimlessly"; "He is a drifter with no direction" |
| **STARTING A PURPOSEFUL ACTION IS STARTING OUT FOR A DESTINATION** | 30 | departure → beginning | "Apply for admission — a journey of a thousand miles…" |
| **THE END OF ACTION IS THE END OF THE PATH** | 30 | arrival → completion | "We are reaching the end of the house-buying process" |
| **LONGTERM PURPOSEFUL ACTIVITY IS A JOURNEY** | 36 | journey → long-term action | parent of the three below |
| **LIFE IS A JOURNEY** (special case 1) | 36-37 | journey → life | "As we travel down life's path…"; "He just sails through life"; "He's lost his way" |
| **LOVE IS A JOURNEY** (special case 2) | 37 | journey → love relationship | "We've hit a crossroads in this relationship" |
| **A CAREER IS A JOURNEY** (special case 3) | 37-38 | journey → career | "He's hit a crossroads in his career"; "half-way up the corporate ladder" |
| **EXTERNAL EVENTS AFFECTING PROGRESS ARE FORCES AFFECTING FORWARD MOTION** | 39 | forces on motion → external events | "He was blown about by the winds of war" |
| **THE PROGRESS OF EXTERNAL EVENTS IS FORWARD MOTION** | 39 | forward motion → events progressing | "Things came to a standstill"; "Let's get the ball moving"; "Let's put the brakes on development" |
| **(EVALUATIVE) COMPARISON OF STATES IN A DYNAMIC SITUATION IS COMPARISON OF DISTANCE / Being Better is Being Farther Along on a Path** | 57 | scale, path → comparison | "He is way ahead of the rest of us"; "We have fallen behind the Soviet Union"; "We were left behind" |
| **LINEAR SCALES ARE PATHS** | 64 | paths → scales | "Bob was ahead of Sam on the scale of intelligence"; "The interest rate fell behind the expected amount" |
| **TIME IS SOMETHING MOVING TOWARD YOU** | 76 | mover → time | "Thursday passed"; "Three o'clock is approaching"; "The witching hour is near" |
| **TIME IS SOMETHING MOVING (No Reference Point)** | 76 | motion → time | "Time crept along"; "Time flies" |
| **FORESEEABLE FUTURE EVENTS ARE UP / AHEAD** | 76 | up, ahead → future | "Upcoming events"; "What's coming up this week?"; "What events are up ahead?" |
| **TIME IS A LANDSCAPE WE MOVE THROUGH** (related metaphor noted on Time page) | 76 | landscape we traverse → time | the mover-ego variant; companion to TIME-MOVES-TOWARD-YOU |
| **MORALITY IS STRAIGHTNESS / MORALITY IS A STRAIGHT PATH** | 185 | straightness, path, straight path → morality | "He's gone straight"; "He's on the straight and narrow path"; "He is a deviant"; "She has strayed"; "He's a crooked businessman" |

Notable additional path-motion submetaphors: A Stage in an Action is a Location Along a Path (p 28); Progress is Measured in Distance to Destination (p 29); Speed of Progress is Speed of Motion to a Destination (p 29); Difficulties are Impediments to Travel (p 37); Obstacles to Action are Obstacles to Motion (p 31); Continuing to Act Despite Difficulty is Moving Despite Obstacles (p 32); Caused Inability to Act is Prevention of Motion (p 32); Opportunities are Open Paths (p 8, 69).

---

## Report 5: EXISTENCE / CREATION / CHANGE

### Canonical Lakoff metaphors for existence-change

**Existence file (pp. 70-71)**
- **EXISTENCE IS A LOCATION (HERE)** / alt. EXISTING IS BEING PRESENT HERE (p. 70). Sub-case of STATES ARE LOCATIONS. Examples: *came about, came into existence, went out of existence, the baby is due, new arrival, forth-coming*.
- **EXISTENCE IS LOCATION UP HERE** (p. 70). *Something came up; the question doesn't arise; arouse/raise suspicion*. Sub-mapping: **Maintaining Existence is Maintaining Location Up Here** — *kept Zen Buddhism from sinking into non-existence*.
- **EXISTENCE IS LOCATION OUT OF CONTAINER** (p. 70). *The emergence of new sciences*.
- **EXISTENCE IS AN OBJECT** (p. 71). Sub-case of PROPERTIES ARE POSSESSIONS. *I want control over my own existence.*
- **EXISTENCE IS LIFE** (p. 71). *birth/death of my interest; killed their play-off hopes; terminally ill program*. (This is the closest named metaphor to BIRTH IS BEGINNING / DEATH IS END.)
- **EXISTENCE IS VISIBILITY** (p. 71). *appearing, vanished, disappear, faded away*.
- **EXISTENCE IS HAVING A FORM** (p. 71). *Things materialized*.

**Creation file (pp. 72-73)**
- **CREATING IS MOVING TO A LOCATION (HERE)** (p. 72). *brought into existence, brought into being*. Inverse of departing from existence.
- **CREATING IS GIVING AN OBJECT** (p. 72). *gave the field its very existence*.
- **CREATING IS MAKING VISIBLE** (p. 72). *erased … by losses; making new problems appear*.
- **CREATING IS BIRTHING** / alt. CAUSATION IS PROGENERATION (p. 72). *gives birth to new problems*.
- **CREATION IS CULTIVATION** (p. 73). *planted seeds, root of all evil, off-shoot, fertile ground*. Implicit destruction-pole: uprooting, withering.
- **CREATING IS MAKING** (p. 73). *made new problems take shape*.

**Related metaphors elsewhere**
- **STATES ARE LOCATIONS** (States, p. 8) — parent of Existence-is-Location; *He is in love; she can stay/remain*. Special case 2 explicitly: **Existence is a Location (Here)**.
- **CHANGE IS MOTION** (Change, p. 15). Includes sub-metaphor **Stopping Being in a State is Leaving a Location** (*came out of the coma*) — the structural template for DEATH IS DEPARTURE.
- **CHANGE IS REPLACEMENT** (Change, p. 18). *was gone, and in its place was…*; useful for DISSOLUTION framed as substitution.
- **ACTIVE IS ALIVE** (Action, p. 26). *alive / dead, lively, liven*.
- **FAILURE IS DEATH / SUCCESS IS LIFE** (Competition, p. 67). *died on that test; team is still alive; killed her*. Cross-references EXISTENCE IS LIFE.
- **SOCIETY IS A BODY** (Society, p. 211). Explicit mapping table: **birth → inception, death → collapse**. *Social death; nation in a depression; social paralysis*.

### Methodological note (honesty for the writeup)

Lakoff does **not** posit a primary image-schema called EXISTENCE/NON-EXISTENCE. Coming-into-being and going-out-of-being are handled as **derived special cases of STATES ARE LOCATIONS** (existence as a here/up/in-container location you arrive at, depart from, or maintain) and as **causation metaphors** in the Creation file (creating-is-moving-to-a-location, creating-is-giving, creating-is-birthing, creating-is-cultivating, creating-is-making). The BIRTH/DEATH pair is attested only via EXISTENCE IS LIFE (p. 71), CREATING IS BIRTHING (p. 72), FAILURE IS DEATH / SUCCESS IS LIFE (p. 67), and the SOCIETY IS A BODY mapping (p. 211).

---

## Report 6: FORCE / BALANCE / DIFFICULTY-BURDEN

### Canonical metaphors

**FORCE-related (causes, influence, logic)**
- **CAUSED CHANGE IS FORCED MOTION** — p 20 (Causation location case). Source: motion/control. "I pushed him into washing the dishes," "Circumstances drove him to attempt suicide," "A sharp word sent her back."
- **CAUSES ARE FORCES (Causes Move Properties to/from Affected Parties)** — p 23 (Causation object case). "The news brought me little comfort," "Years of abuse took away her smile."
- **FORCE IS A SUBSTANCE DIRECTED AT AN AFFECTED PARTY** — p 170. "Apply more force," "Exert more force," "Put more force behind your punches."
- **FORCE IS A SUBSTANCE CONTAINED IN AFFECTING CAUSES** — p 171. "He said some forceful words," "The force of the blow knocked me over."
- **A FORCE IS A MOVING OBJECT** — p 172. Object hits/holds/moves affected party; strength of object determines its ability to affect.
- **EXTERNAL EVENTS AFFECTING PROGRESS ARE FORCES AFFECTING FORWARD MOTION** — p 39. Beneficial events are forces moving toward your destination; detrimental ones are opposing forces.
- **DESIRES THAT CONTROL ACTION ARE EXTERNAL FORCES THAT CONTROL MOTION** — p 33. "That coat pulled me into the store."
- **PSYCHOLOGICAL FORCES ARE PHYSICAL FORCES (Influence is a Force; Manipulation is Physical Manipulation)** — p 131. "He can exert his influence," "She could bend his will," "Reagan brought pressure to bear."
- **LOGIC IS CAUSATION IS CONTROL OVER MOTION / LOGIC IS A FORCE THAT MOVES A MIND** — p 125. "His reasoning forced me to a conclusion," "Your premises aren't going to allow you to reach the correct solution."
- **LOGIC IS GRAVITY** — p 126. Theory not adequately supported topples under the force of logic.
- **OBLIGATIONS ARE FORCES** — p 207. "I was forced to do it."
- **Control over Action is Control over Motion** — p 31 (Action). "He pushed me into doing it," "She held him back," "She has her secretary on a tight rein."

**BALANCE-related**
- **Comparison of Importance is Weighing** — p 62. "You have to weigh the pros and cons," "The scale just tipped in favor."
- **MENTAL ACCOUNTING / Results are Net Balances** — p 136. Source domain accounting; "to sum it all up," "the bottom line."
- **MORAL ACCOUNTING** — p 188. Source domain: "debt, accounting, balance"; deeds as credit/debt, paying off, settling.
- **Compliance is Tightness** (vs slackness) — p 208. "Tight ship" / "slack ship."

**DIFFICULTIES-related (burden, obstacle, support, opponent)**
- **DIFFICULTIES ARE CONTAINERS** — p 75. "In hot water," "in a mess," "get out of this situation."
- **Obstacles to Action are Obstacles to Motion** — p 31. "Roadblock," "impasse," "brick wall," "uphill battle," "hurdles."
- **Continuing to Act Despite Difficulty is Moving Despite Obstacles** — p 32. "Made it through the rough spots."
- **Caused Inability to Act is Prevention of Motion** — p 32. "Regulations keep me from moving ahead."
- **Aids to Action are Aids to Motion** — p 32. "My Dad made the way smooth."
- **Easy Action is Easy Motion** — p 31. "Smooth sailing," "all downhill."
- **Difficulties are Impediments to Travel** — p 37 (Longterm Action / Life is a Journey). "He's lost his way," "rocky road ahead."
- **OBLIGATIONS ARE BURDENS (on shoulder/back)** — p 204 (Responsibilities). "Loaded with responsibilities," "shouldered the task," "weighed down with obligations," "carrying a heavy load," "pressing obligation," "bears the responsibility."
- **RESPONSIBILITIES ARE BURDENS** — p 206. "Forced to bear the blame," "left holding the bag," "saddled with the blame."
- **Believers/Supporters; Theories Need Support (collapse)** — p 119 (Debate, THEORIES ARE BUILDINGS): "uphold," "supporter," "undermine," "topple," "foundations." Linked to LOGIC IS GRAVITY.
- **Difficulty is Hardness of Object** — p 195 (Problems). "Hard problem," "impenetrable."

### Note on caveats

"light" in DIFFICULTY-BURDEN is intentionally the weight sense (cf. "heavy load" p 204), not illumination — Lakoff's LIGHT/DARK file is at p 177 and uses different examples. All pairs are single tokens. Some pairs use slightly modernized antonyms (e.g., "wobbly," "askew") where Lakoff's examples ground the schema but don't supply a clean single-word opposite.
