"""Final SVG post-processing helpers."""


def modify_svg(full_svg_str: str) -> str:
    word_svg = (
        '<path fill="none" stroke="#000" stroke-width="4" d="M342 342 H354 M348 342 V356"/>'
        '<path fill="none" stroke="#fff" stroke-width="2" d="M343 342 H353 M348 342 V355"/></svg>'
    )
    return full_svg_str.replace("</svg>", word_svg)
