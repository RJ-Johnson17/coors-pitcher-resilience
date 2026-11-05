"""
Keeps the Streamlit app awake by visiting it in a real browser.

Run automatically every few hours by .github/workflows/keep_awake.yml.
If the app is asleep, it clicks the wake-up button and waits for the app to load.
"""

from playwright.sync_api import sync_playwright

APP_URL = "https://coors-pitchers.streamlit.app"
WAKE_TEXT = "get this app back up"


def click_wake_button(page) -> bool:
    """Look for the wake-up button on the page and in any frames; click it if found."""
    for frame in [page.main_frame, *page.frames]:
        button = frame.get_by_role("button", name=WAKE_TEXT, exact=False)
        if button.count() > 0:
            button.first.click()
            return True
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(15_000)  # give the page time to show either the app or the sleep screen

        if click_wake_button(page):
            print("App was asleep. Clicked the wake-up button.")
            page.wait_for_timeout(90_000)  # app takes a minute or so to start back up
        else:
            print("App was already awake.")
            page.wait_for_timeout(20_000)  # stay a little while so the visit counts

        browser.close()


if __name__ == "__main__":
    main()
