from playwright.sync_api import sync_playwright

url = 'http://localhost:8000/thin-lens-sandbox.html'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 980})
    page.on('pageerror', lambda exc: print('PAGEERROR:', exc))
    page.goto(url, wait_until='networkidle', timeout=30000)
    page.screenshot(path='thin-lens-sandbox.png', full_page=True)
    print('TITLE:', page.title())
    print('URL:', page.url)
    print('TEXT_OK:', 'Dünne Linse' in page.text_content('body'))
    browser.close()
