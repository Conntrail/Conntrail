GOAL:    The viewer understands, in one sentence, that Conntrail measures how
         close an agent's routing decision was to flipping — and clicks the live demo.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps.
TIME:    30 s = 900 frames. No music source is available (no HeyGen, no local
         MusicGen), so this cut is silent; captions carry it. Music can be added
         later with the media-use skill.

LOOK:    ground #f4f7fb with a 28 px graph grid at 5% ink, softened by a radial
         falloff; ink #10151d; muted #556070; border #dfe6f0; accent #1d4ed8,
         rationed to one element per shot; stability scale #15803d / #b45309 /
         #dc2626 only for verdicts. Display Inter 900, 64–84 px, -0.02em;
         body Inter 400/600, 22–30 px; data JetBrains Mono 500, 20–26 px.
         Medium: product UI cards on a light sweep — white, 1 px border, 14 px
         radius, 0 18px 40px rgba(16,24,40,.10). Light from upper left; no grain,
         no vignette.
OBJECTS: support chat card (header, bubbles, route chip), route rows, entropy
         gauge (SVG dial, ticks at 0/.25/.5/.75, thresholds at .25/.60),
         decision spectrum, attribution line, mark (one node splitting into three).
MOTION:  arrivals expo.out 0.5–0.7 s; gauge sweep power2.inOut 1.2 s; chip flip
         power3.inOut 0.45 s; exits power3.in 0.2 s; staggers 70–110 ms.
         One hero move per shot.
SHOTS:   
  1. 0.0–6.0  hook: chat card in; customer message; bot reply; policy_answer chip.
              Reads: "The bot answered with confidence."
  2. 6.0–11.0 flip: neutral paraphrase; chip flips to escalate; route rows.
              Reads: "Same request. Different route."
  3. 11.0–16.5 gauge: dial sweeps to 0.75 fragile.
              Reads: "Conntrail measures the distance between them."
  4. 16.5–24.5 proof: decision spectrum (similar / neutral / opposite), one flip
              highlighted; attribution "semantic intensity"; counterfactual "decline".
              Reads: "One dimension flipped it."
  5. 24.5–30.0 claim/end: claim line, then mark + CTA.
              Reads: "You can see how close it was to deciding otherwise."
              / "Try the live demo — conntrail.ibzie.dev"

BANNED:  dark void; glow, bloom, particles; purple/blue gradients; cream paper
         with serif and mono labels; three feature cards in a row; docked logo;
         every scene fading and sliding up; titles drifting after they land;
         invented screens not in the product.
QA:      2.0 s chat readable; 8.5 s chip flipped; 14.0 s 0.75 + fragile;
         21.0 s three variants + attribution; 28.5 s end card + CTA.
