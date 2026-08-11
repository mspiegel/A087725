#!/usr/bin/env python3
"""Draw the README's 8-puzzle figure: a smiley face sliced across the tiles.

Two boards side by side. Each tile carries the ninth of the face that belongs
at that tile's *home* square, so the solved board assembles the face (minus the
blank) and the scrambled board breaks it up.

The boards are not invented. They are the goal and one of the two depth-31
antipodes that `tests/invariants.rs` asserts, so the figure cannot drift from
what the code says:

    GOAL     = 1 2 3 / 4 5 6 / 7 8 _    src/puzzle8/state.rs
    ANTIPODE = 8 6 7 / 2 5 4 / 3 _ 1    tests/invariants.rs

Output is a single self-contained SVG with its own background, legible under
both GitHub themes. Three constraints come from how GitHub renders SVG:

  * it renders only as a referenced file (`![](docs/x.svg)`), never as inline
    `<svg>` in the markdown, which is stripped;
  * `@import`ed web fonts do not load, so the numerals use generic sans-serif;
  * GitHub's sanitizer has been reported to strip `dominant-baseline`, so text
    is centred with an explicit `dy` instead.

Usage:  python3 scripts/make_puzzle_figure.py > docs/eight-puzzle.svg
"""

import sys

# --- the two boards, row-major, 0 = blank -----------------------------------
GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0]
ANTIPODE = [8, 6, 7, 2, 5, 4, 3, 0, 1]

# --- geometry ---------------------------------------------------------------
CELL = 60
BOARD = 3 * CELL
MARGIN = 22
BOARD_GAP = 56
CAPTION_H = 26
INSET = 2.0  # tile gap, so tiles read as separate pieces
RADIUS = 7

W = MARGIN * 2 + BOARD * 2 + BOARD_GAP
H = MARGIN + BOARD + CAPTION_H + MARGIN

# --- palette: fixed, chosen to read on both light and dark GitHub -----------
BG = "#eef1f5"
TILE = "#ffffff"
TILE_BLANK = "#dfe4ea"
EDGE = "#b6c0cc"
YELLOW = "#ffd23f"
INK = "#2b2b2b"
CAPTION = "#3d4650"


def face(dx, dy):
    """The smiley, drawn in board-local coordinates, shifted by (dx, dy).

    Primitives only — no emoji glyph. A text emoji would render differently on
    every platform, and may not render at all inside an <img>-loaded SVG.
    """
    c = BOARD / 2
    return (
        f'<g transform="translate({dx:g},{dy:g})">'
        f'<circle cx="{c:g}" cy="{c:g}" r="{c - 18:g}" fill="{YELLOW}" '
        f'stroke="{INK}" stroke-width="3"/>'
        f'<circle cx="{c - 22:g}" cy="{c - 20:g}" r="9" fill="{INK}"/>'
        f'<circle cx="{c + 22:g}" cy="{c - 20:g}" r="9" fill="{INK}"/>'
        f'<path d="M {c - 30:g} {c + 14:g} Q {c:g} {c + 46:g} {c + 30:g} {c + 14:g}" '
        f'fill="none" stroke="{INK}" stroke-width="7" stroke-linecap="round"/>'
        f"</g>"
    )


def board(board_state, ox, oy, tag, caption):
    """One 3x3 board at (ox, oy). Each tile clips its own window of the face."""
    out = []
    for pos, tile in enumerate(board_state):
        px, py = (pos % 3) * CELL, (pos // 3) * CELL
        x, y = ox + px + INSET, oy + py + INSET
        side = CELL - 2 * INSET

        if tile == 0:
            out.append(
                f'<rect x="{x:g}" y="{y:g}" width="{side:g}" height="{side:g}" '
                f'rx="{RADIUS}" fill="{TILE_BLANK}" stroke="{EDGE}" '
                f'stroke-width="1.5" stroke-dasharray="5 4"/>'
            )
            continue

        # The slice this tile owns is the one at its home square in the goal.
        home = tile - 1
        hx, hy = (home % 3) * CELL, (home // 3) * CELL
        cid = f"clip-{tag}-{pos}"

        out.append(
            f'<clipPath id="{cid}">'
            f'<rect x="{x:g}" y="{y:g}" width="{side:g}" height="{side:g}" rx="{RADIUS}"/>'
            f"</clipPath>"
            f'<g clip-path="url(#{cid})">'
            f'<rect x="{x:g}" y="{y:g}" width="{side:g}" height="{side:g}" fill="{TILE}"/>'
            f"{face(ox + px - hx, oy + py - hy)}"
            f"</g>"
            f'<rect x="{x:g}" y="{y:g}" width="{side:g}" height="{side:g}" '
            f'rx="{RADIUS}" fill="none" stroke="{EDGE}" stroke-width="1.5"/>'
            # Numeral on a backing disc: the face art sits under it, and a bare
            # glyph collides with an eye on the centre tile. `dy` centres the
            # text without dominant-baseline, which GitHub's sanitizer strips.
            f'<circle cx="{x + 13:g}" cy="{y + side - 13:g}" r="9.5" fill="{TILE}" '
            f'opacity="0.88"/>'
            f'<text x="{x + 13:g}" y="{y + side - 13:g}" dy="4.6" '
            f'font-family="sans-serif" font-size="13" font-weight="bold" '
            f'fill="{INK}" text-anchor="middle">{tile}</text>'
        )

    out.append(
        f'<text x="{ox + BOARD / 2:g}" y="{oy + BOARD + 18:g}" dy="0" '
        f'font-family="sans-serif" font-size="14" fill="{CAPTION}" '
        f'text-anchor="middle">{caption}</text>'
    )
    return "".join(out)


def main():
    left = MARGIN
    right = MARGIN + BOARD + BOARD_GAP
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="An 8-puzzle solved and scrambled, with a smiley face '
        f'split across the tiles">',
        f'<rect width="{W}" height="{H}" rx="10" fill="{BG}"/>',
        board(GOAL, left, MARGIN, "s", "solved"),
        board(ANTIPODE, right, MARGIN, "x", "scrambled"),
        "</svg>",
    ]
    sys.stdout.write("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
