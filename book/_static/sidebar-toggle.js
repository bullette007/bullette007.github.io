// Support every sidebar button, including the article-header duplicates.
document.addEventListener('click', function (event) {
  const button = event.target.closest('button.sidebar-toggle');
  if (!button) return;
  const primary = button.classList.contains('primary-toggle');
  const selector = primary ? '.primary-toggle' : '.secondary-toggle';
  const desktop = window.matchMedia(primary ? '(min-width: 992px)' : '(min-width: 1200px)').matches;
  if (desktop) {
    const sidebar = document.getElementById(primary ? 'pst-primary-sidebar' : 'pst-secondary-sidebar');
    if (!sidebar) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const hidden = sidebar.classList.toggle('ci-sidebar-hidden');
    document.querySelectorAll(selector).forEach(toggle => toggle.setAttribute('aria-expanded', String(!hidden)));
  } else {
    // Let the theme open its native mobile dialog using its registered button.
    const first = document.querySelector(selector);
    if (first && first !== button) {
      event.preventDefault();
      event.stopImmediatePropagation();
      first.click();
    }
  }
}, true);
