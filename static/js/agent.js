/* Agent-mode UI helpers (pure functions, used by static/index.html):
   step icons/status labels and a safe unified-diff renderer. */

/** Escape text for HTML interpolation. */
function escHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

/** Lucide icon name for an agent tool. */
function stepIcon(tool) {
  switch (tool) {
    case 'list_dir': return 'folder-tree';
    case 'read_file': return 'file-text';
    case 'grep': return 'search';
    case 'write_file': return 'file-plus';
    case 'edit_file': return 'pencil';
    case 'run_shell': return 'terminal';
    default: return 'wrench';
  }
}

/** Human label for a step status. */
function stepStatusLabel(status) {
  switch (status) {
    case 'running': return 'running';
    case 'pending_approval': return 'waiting approval';
    case 'approved': return 'approved';
    case 'ok': return 'ok';
    case 'error': return 'error';
    case 'denied': return 'denied';
    default: return status || '';
  }
}

/**
 * Render a unified diff as colored, escaped HTML lines.
 * Every line is HTML-escaped first (tool/model output is untrusted), and the
 * assembled markup (spans with fixed classes only) is passed through
 * DOMPurify when available as a second layer of defence.
 */
function renderUnifiedDiff(diff) {
  const lines = String(diff || '').split('\n');
  // Drop a single trailing empty line from the final newline.
  if (lines.length && lines[lines.length - 1] === '') lines.pop();
  const html = lines.map(line => {
    let cls = 'd-ctx';
    if (line.startsWith('+++') || line.startsWith('---')) cls = 'd-file';
    else if (line.startsWith('@@')) cls = 'd-hunk';
    else if (line.startsWith('+')) cls = 'd-add';
    else if (line.startsWith('-')) cls = 'd-del';
    return `<span class="${cls}">${escHtml(line) || ' '}</span>`;
  }).join('');
  return (typeof DOMPurify !== 'undefined') ? DOMPurify.sanitize(html) : html;
}
