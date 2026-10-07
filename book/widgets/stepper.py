"""Display an automatically sized image-sequence iframe in classic Jupyter/RISE or Jupyter Book."""

import html
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from IPython.display import HTML, display


def display_stepper(src, **params):
    """Embed a widget URL, optionally adding/overriding its query parameters.

    The notebook must be trusted to run the output's resizing script.
    The widget and its images must be served at ``src`` in each environment.
    """
    parts = urlsplit(src)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault('ui', 'minimal')
    query.update({key: str(value) for key, value in params.items()})
    query['resize'] = 'auto'
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    frame_id = 'ci-stepper-' + uuid4().hex
    display(HTML(
        f'<iframe id="{frame_id}" src="{html.escape(url, quote=True)}" '
        'title="Interactive image sequence" '
        'style="display:block;width:100%;aspect-ratio:16/9;border:0" '
        'scrolling="no"></iframe>'
        '<script>(function () {'
        f'const frame = document.getElementById({json.dumps(frame_id)});'
        'let pending = false;'
        'function updateLimit() {'
        'pending = false;'
        'if (!frame.isConnected) return;'
        'const presenting = document.body.classList.contains("rise-enabled") || '
        'document.body.classList.contains("reveal-viewport");'
        'const slide = frame.closest(".reveal section.present") || frame.closest(".reveal section");'
        'let height = null;'
        'if (presenting && slide && frame.offsetWidth) {'
        'const rect = frame.getBoundingClientRect();'
        'const scale = rect.width / frame.offsetWidth;'
        'const viewport = document.querySelector(".reveal").getBoundingClientRect();'
        'const viewportHeight = Math.min(window.innerHeight, viewport.height || window.innerHeight);'
        'const before = Math.max(0, rect.top - slide.getBoundingClientRect().top);'
        'height = Math.max(180, Math.floor((viewportHeight * 0.9 - before) / (scale || 1)));'
        '}'
        'frame.contentWindow.postMessage({type:"ci-stepper-limit", height}, "*");'
        '}'
        'function scheduleLimit() {'
        'if (!pending) { pending = true; requestAnimationFrame(updateLimit); }'
        '}'
        'function resize(event) {'
        'if (!frame.isConnected) { window.removeEventListener("message", resize); return; }'
        'if (event.source !== frame.contentWindow || event.data?.type !== "ci-stepper-height") return;'
        'const height = event.data.height;'
        'if (typeof height !== "number" || !Number.isFinite(height) || height <= 0) return;'
        'frame.style.height = Math.ceil(height) + "px";'
        'frame.style.aspectRatio = "auto";'
        '}'
        'window.addEventListener("message", resize);'
        'frame.addEventListener("load", function () {'
        'frame.contentWindow.postMessage({type:"ci-stepper-measure"}, "*");'
        'scheduleLimit();'
        '});'
        'window.addEventListener("resize", scheduleLimit);'
        'let lastWidth = 0;'
        'new ResizeObserver(function () {'
        'if (frame.offsetWidth !== lastWidth) { lastWidth = frame.offsetWidth; scheduleLimit(); }'
        '}).observe(frame);'
        'new MutationObserver(function (records) {'
        'if (records.some(r => r.target === document.body || '
        'r.target.matches(".reveal, .reveal .slides, .reveal section"))) scheduleLimit();'
        '}).observe(document.body, {attributes:true, subtree:true, attributeFilter:["class", "style"]});'
        'scheduleLimit();'
        '})();</script>'
    ))
