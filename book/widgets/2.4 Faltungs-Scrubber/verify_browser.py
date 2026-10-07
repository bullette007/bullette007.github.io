"""Numerical and interaction checks against the standalone file:// widget."""
from pathlib import Path
from math import isclose
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = (ROOT / "index.html").as_uri()


def diagnostics(page):
    page.wait_for_timeout(200)
    return page.evaluate("window.__convolutionDiagnostics()")


def near(actual, expected):
    assert len(actual) == len(expected)
    assert all(isclose(a, b, abs_tol=1e-10) for a, b in zip(actual, expected))


def set_range(page, selector, value):
    page.locator(selector).evaluate(
        "(el, value) => { el.value=value; el.dispatchEvent(new Event('input', {bubbles:true})); }",
        value,
    )


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1100})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(URL)
    d = diagnostics(page)
    assert d["state"]["lang"] == "en"
    assert page.locator("html").get_attribute("lang") == "en"
    assert page.title() == "2.4 · Convolution Scrubber"
    assert page.locator("#f option[value=exp]").text_content() == "One-sided exponential pulse"
    page.locator("#language").click()
    d = diagnostics(page)
    assert d["state"]["lang"] == "de" and "lang=de" in page.url
    assert page.locator("#clearF").text_content() == "f leeren"
    page.goto(page.url)
    assert diagnostics(page)["state"]["lang"] == "de"
    page.locator("#mode2").click()
    diagnostics(page)
    assert "Σ der 9 Produkte" in page.locator("#sum").text_content()
    page.locator("#language").click()
    diagnostics(page)
    assert "Sum of the 9 products" in page.locator("#sum").text_content()
    assert page.locator("#kernelType option[value=asymmetric]").text_content() == "Asymmetric"
    page.locator("#mode1").click()
    diagnostics(page)
    # Independently form the full discrete convolution, including negative values.
    page.locator("#f").select_option("bipolar")
    d = diagnostics(page)
    expected = [0.0] * 481
    for i, f in enumerate(d["f"]):
        for j, h in enumerate(d["h"]):
            expected[i + j] += f * h * d["dx"]
    near(d["g"], expected)
    page.locator("#boxes").click()
    d = diagnostics(page)
    assert isclose(d["g"][240], 1.975, abs_tol=1e-10)
    near(d["g"], list(reversed(d["g"])))
    page.locator("#identity").click()
    d = diagnostics(page)
    near(d["g"][120:361], d["f"])
    page.locator("#response").click()
    d = diagnostics(page)
    near(d["g"][120:361], d["h"])
    # Native range input and global keyboard navigation.
    set_range(page, "#x", 1)
    page.locator("h1").click()
    page.keyboard.press("ArrowRight")
    assert diagnostics(page)["state"]["x"] == 1.05
    page.keyboard.press("Home")
    assert diagnostics(page)["state"]["x"] == -12
    page.keyboard.press("Space")
    page.wait_for_timeout(300)
    page.keyboard.press("Space")
    assert diagnostics(page)["state"]["x"] > -12
    # Drawing interpolates between pointer samples and survives URL round-tripping.
    page.locator("#clearF").click()
    rect = page.locator("#sourceF").bounding_box()
    page.mouse.move(rect["x"] + rect["width"] * .25, rect["y"] + rect["height"] * .3)
    page.mouse.down()
    page.mouse.move(rect["x"] + rect["width"] * .7, rect["y"] + rect["height"] * .6, steps=12)
    page.mouse.up()
    d = diagnostics(page)
    assert d["state"]["f"] == "free"
    assert max(d["f"]) > .3 and min(d["f"]) < -.1
    saved = page.url
    page.goto(saved)
    restored = diagnostics(page)
    assert all(abs(a-b) <= .0051 for a,b in zip(d["f"], restored["f"]))
    # Convolution orientation: right-of-center kernel displaces an impulse right.
    page.locator("#mode2").click()
    page.locator("#imageType").select_option("impulse")
    page.locator("#kernelType").select_option("asymmetric")
    d = diagnostics(page)
    assert d["result2"][6 * 12 + 6] == .25
    assert d["result2"][6 * 12 + 7] == .75
    assert sum(d["result2"]) == 1
    # Compare every pixel against the direct definition, including Sobel signs and edges.
    page.locator("#kernelType").select_option("sobel")
    page.locator("#imageType").select_option("shapes")
    d = diagnostics(page)
    expected = []
    for m in range(12):
        for n in range(12):
            value = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= m-i < 12 and 0 <= n-j < 12:
                        value += d["image"][(m-i)*12+n-j] * d["kernel"][(i+1)*3+j+1]
            expected.append(value)
    near(d["result2"], expected)
    assert min(expected) < 0 < max(expected)
    page.locator("#kernelType").select_option("box")
    set_range(page, "#pixel", 0)
    d = diagnostics(page)
    assert d["detail"]["patch"][:3] == [0, 0, 0]
    assert d["detail"]["patch"][3] == 0
    # Coefficients, selections, theme and full-result state are shareable.
    page.locator("#kernelInputs input").nth(0).fill("-0.5")
    page.locator("#full").check()
    page.locator("#theme").click()
    d = diagnostics(page)
    page.goto(page.url)
    restored = diagnostics(page)
    assert restored["state"] == d["state"]
    near(restored["kernel"], d["kernel"])
    page.locator("#reset").click()
    d = diagnostics(page)
    assert d["state"]["mode"] == "2d" and d["state"]["theme"] == "dark"
    assert d["state"]["kernelType"] == "gauss" and not d["state"]["full"]
    page.goto(URL + "?mode=2d&x=invalid&pixel=9999&kernel=oops")
    d = diagnostics(page)
    assert d["state"]["pixel"] == 143 and d["state"]["x"] == .5
    # Both responsive modes must fit the viewport without horizontal scrolling.
    page.set_viewport_size({"width": 390, "height": 844})
    for selector in ["#mode1", "#mode2"]:
        page.locator(selector).click()
        diagnostics(page)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.set_viewport_size({"width": 1500, "height": 1100})
    page.goto(URL)
    diagnostics(page)
    page.screenshot(path=str(ROOT / "preview.png"), full_page=True)
    assert not errors, errors
    browser.close()
    print("PASS: reference convolutions, impulses, drawing, URL state, controls, mobile layout; preview.png generated.")
