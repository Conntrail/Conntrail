# REVIEW — Conntrail explainer (30 s)

## Creative calls
- **Built storyline A ("The flip").** One support message, the bot answers
  confidently, a neutral paraphrase flips the route, the entropy gauge makes the
  distance visible. Chosen from the research: the strongest YC/devtools demos
  open on the problem or the product working, follow one workflow, and hold long
  enough to read.
- **The chat is live, not a slideshow.** The customer's message is typed into the
  input and sent; the bot shows a typing indicator then answers; the customer
  then spams three short messages out of frustration ("Are you sure?", "The
  booking says non-refundable.", "I need this fixed now."). The flip scene types
  and sends its paraphrase the same way, and the route chip flips in reaction.
- **Not built:** B ("Four incidents", a breadth montage — research warns it
  dilutes a 30 s film) and C ("The instrument", an abstract device — the
  explainer guidance says don't open on the device alone). Both remain options.
- **Real product UI, rebuilt faithfully.** The HyperFrames website capture was
  blocked by an access-protection response, so the chat card, route chips,
  gauge, spectrum and copy were rebuilt from the site's own markup/CSS and the
  real fixture data (`support-chatbot-policy`: 0.75, fragile, semantic
  intensity, counterfactual decline). Nothing is invented.
- **Copy is the product's own:** the customer message, the bot reply, the three
  contrast variants, the route names and the counterfactual come from the demo
  fixture. The claim line is the positioning sentence.
- **Timing:** hook 0–8 s, flip 8–13 s, gauge 13–18 s, proof 18–24 s, claim 24–30 s.

## Measured vs looked at
- Measured: `check` (lint, runtime, layout, contrast, motion) passes with one
  warning; timeline and clip windows; text floor and contrast; snapshot sheets
  (beat, seam, phone).
- Looked at: the beat sheet, the 6 s cut, the claim and end card, the 390 px
  phone sheet.
- **Sound: none.** No music source was available (no HeyGen sign-in, no local
  MusicGen), so the film is silent by the skill's fallback. It was not heard.

## Weakest shot
- The 6 s cut into the flip originally landed on an empty card; fixed by making
  the paraphrase and the escalated reply visible at scene start.
- With more time: the hook card has empty space below the exchange — bringing
  the input row up or shortening the card would tighten it. The end-card mark
  could be larger.

## Warnings kept on purpose
- `film_silent` / `audio_none` — deliberate. Add music later with the media-use
  skill (catalog via HeyGen, or local MusicGen): a minimal pulse ~110 bpm with a
  tick on each route flip and one impact on the gauge. Do not synthesize it in
  code.

## Open
- Approval to render. The Studio preview is the review surface.
