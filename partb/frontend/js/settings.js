(function () {
  'use strict';

  /* ================================================================
     CUSTOM FORM DROPDOWN — Fixed position, escapes all overflow
  ================================================================ */

  function initCustomFormDropdown(dropdownId, toggleId, labelId, selectId, menuId) {
    var dropdown = document.getElementById(dropdownId);
    var toggle   = document.getElementById(toggleId);
    var label    = document.getElementById(labelId);
    var select   = document.getElementById(selectId);
    var menu     = document.getElementById(menuId);
    if (!dropdown || !toggle || !menu) return;

    var items  = menu.querySelectorAll('.dropdown-item');
    var isOpen = false;

    /* Move menu to <body> so it escapes every overflow/clip ancestor */
    document.body.appendChild(menu);
    menu.style.position  = 'fixed';
    menu.style.zIndex    = '99999';
    menu.style.transform = 'none';
    menu.style.transition = 'opacity 0.15s, visibility 0.15s';

    function positionMenu() {
      var rect = toggle.getBoundingClientRect();
      menu.style.top    = (rect.bottom + 4) + 'px';
      menu.style.left   = rect.left + 'px';
      menu.style.width  = rect.width + 'px';
      menu.style.bottom = 'auto';
      menu.style.right  = 'auto';
    }

    function openMenu() {
      positionMenu();
      dropdown.classList.add('open');
      menu.classList.add('open');
      isOpen = true;
    }

    function closeMenu() {
      dropdown.classList.remove('open');
      menu.classList.remove('open');
      isOpen = false;
    }

    toggle.addEventListener('click', function (e) {
      e.stopPropagation();
      if (isOpen) { closeMenu(); } else { openMenu(); }
    });

    document.addEventListener('click', function (e) {
      if (isOpen && !toggle.contains(e.target) && !menu.contains(e.target)) {
        closeMenu();
      }
    });

    window.addEventListener('scroll', function () { if (isOpen) positionMenu(); }, true);
    window.addEventListener('resize', function () { if (isOpen) positionMenu(); });

    items.forEach(function (item) {
      item.addEventListener('click', function (e) {
        e.stopPropagation();
        items.forEach(function (i) { i.classList.remove('selected'); });
        item.classList.add('selected');

        var titleEl = item.querySelector('.book-item-title') || item.querySelector('span');
        var text    = titleEl ? titleEl.textContent.trim() : item.getAttribute('data-value');

        select.value        = item.getAttribute('data-value');
        label.textContent   = text;
        label.style.color   = 'var(--ink)';

        closeMenu();
      });
    });
  }

  /* ================================================================
     DELETE CONTEXT MODAL
  ================================================================ */

  var deleteContextModal  = document.getElementById('delete-context-modal');
  var deleteContextForm   = document.getElementById('delete-context-form');
  var deleteContextCancel = document.getElementById('delete-context-cancel');
  var deleteContextClose  = document.getElementById('delete-context-close');

  function openDeleteContextModal() {
    if (!deleteContextModal) return;
    var select = document.getElementById('delete-book-select');
    if (select) select.value = '';
    var label = document.getElementById('delete-book-label');
    if (label) { label.textContent = 'Choose a book...'; label.style.color = 'var(--mute)'; }
    document.querySelectorAll('#delete-book-menu .dropdown-item')
            .forEach(function (i) { i.classList.remove('selected'); });
    deleteContextModal.classList.add('open');
  }

  function closeDeleteContextModal() {
    if (deleteContextModal) deleteContextModal.classList.remove('open');
  }

  if (deleteContextClose) {
    deleteContextClose.addEventListener('click', closeDeleteContextModal);
  }

  if (deleteContextCancel) {
    deleteContextCancel.addEventListener('click', closeDeleteContextModal);
  }

  if (deleteContextModal) {
    deleteContextModal.addEventListener('click', function (e) {
      if (e.target === deleteContextModal) closeDeleteContextModal();
    });
  }

  if (deleteContextForm) {
    deleteContextForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var bookId = document.getElementById('delete-book-select').value;
      if (!bookId) { alert('Please select a book first.'); return; }
      alert('Context for "' + bookId + '" deleted successfully.');
      closeDeleteContextModal();
    });
  }

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeDeleteContextModal();
  });

  /* ================================================================
     DELETE BOOK BUTTONS ON SETTINGS PAGE → open the modal
  ================================================================ */

  document.querySelectorAll('[data-action="delete-context"]').forEach(function (btn) {
    btn.addEventListener('click', openDeleteContextModal);
  });

  /* ================================================================
     THEME TOGGLE
  ================================================================ */

  var themeToggle = document.getElementById('theme-toggle');
  var savedTheme  = localStorage.getItem('krutrim-theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);

  if (themeToggle) {
    themeToggle.innerHTML = savedTheme === 'dark'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';

    themeToggle.addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('krutrim-theme', next);
      themeToggle.innerHTML = next === 'dark'
        ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    });
  }

  /* ================================================================
     INIT
  ================================================================ */

  initCustomFormDropdown(
    'delete-book-dropdown',
    'delete-book-toggle',
    'delete-book-label',
    'delete-book-select',
    'delete-book-menu'
  );

})();
