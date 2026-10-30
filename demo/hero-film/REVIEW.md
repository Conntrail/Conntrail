# REVIEW — Conntrail hero loop (12 s)

## Creative calls
- A single seamless 12 s loop, not a slideshow: the customer types and sends the
  message, the bot answers, a neutral paraphrase is typed and sent, the route
  flips to `escalate`, the gauge sweeps to 0.75 fragile, then it resets to the
  start for the loop.
- Same instrument aesthetic as the site: graph paper, hairlines, mono data,
  stability colours only for verdicts. The chat window is the hero; the
  measurement panel reads the result.
- All copy/data from the real fixture: original → `policy_answer`, neutral →
  `escalate`, opposite → `decline`, 0.75, semantic intensity, counterfactual
  decline.

## Measured vs looked at
- Measured: `lint` 0/0; `check` ok with two deliberate warnings (below).
- Looked at: snapshot sheets at 0.8, 2.8, 4.8, 7.4, 9.2, 11.8 s.
- Sound: none (silent loop, muted autoplay on the site).

## Warnings kept on purpose
- `container_overflow` on the composer: the composer bar sits at the bottom of
  the chat window and the checker's clipping heuristic flags it; visually it is
  inside the frame (see the sheet).
- `line_unread` on the closing "fragile" badge: it is part of a loop, so it
  intentionally clears as the scene resets. The hold before reset is ~2.5 s.

## Open
- None blocking. If the loop read feels tight, raise the hold at 9.3 s or slow
  the reset.
