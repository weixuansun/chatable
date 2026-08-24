/* chatable UI enhancements — global helpers used by static/index.html */

/**
 * Show a short toast notification in the bottom-right corner.
 * @param {string} message
 * @param {'info'|'success'|'error'} type
 */
function toast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);

  // trigger reflow for transition
  requestAnimationFrame(() => el.classList.add('show'));

  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

/**
 * Highlight all <pre><code> blocks inside a rendered markdown element.
 * Guards against double-highlighting.
 */
function highlightCodeBlocks(el) {
  if (typeof hljs === 'undefined') return;
  el.querySelectorAll('pre code').forEach((block) => {
    if (block.dataset.highlighted === 'yes') return;
    hljs.highlightElement(block);
    block.dataset.highlighted = 'yes';
  });
}

/**
 * Add a copy button to every <pre> block inside a rendered element.
 */
function addCodeCopyButtons(el) {
  el.querySelectorAll('pre').forEach((pre) => {
    if (pre.querySelector('.copy-btn')) return;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');

    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      const text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => (btn.textContent = 'Copy'), 1500);
      }).catch(() => {
        btn.textContent = 'Failed';
        setTimeout(() => (btn.textContent = 'Copy'), 1500);
      });
    });

    pre.appendChild(btn);
  });
}

// Global keyboard shortcut: Cmd/Ctrl+K focuses the tree search box.
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    const searchInput = document.querySelector('.tree-search input');
    if (searchInput && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
  }
});
