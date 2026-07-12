import { state, booksMap, escapeHtml, API } from './state.js';

export function initPdfViewer() {
  var overlay       = document.getElementById('pdf-viewer-modal');
  var closeBtn      = document.getElementById('pdf-viewer-close');
  var prevBtn       = document.getElementById('pdf-prev-btn');
  var nextBtn       = document.getElementById('pdf-next-btn');
  var titleEl       = document.getElementById('pdf-viewer-title');
  var subtitleEl    = document.getElementById('pdf-viewer-subtitle');
  var badgeEl       = document.getElementById('pdf-page-badge');
  var curPageEl     = document.getElementById('pdf-current-page');
  var totPageEl     = document.getElementById('pdf-total-pages');
  var canvas        = document.getElementById('pdf-render-canvas');
  var resizeHandle  = overlay ? overlay.querySelector('.pdf-drawer-handle') : null;
  if (!overlay) return;

  var MIN_WIDTH = 320;
  var MAX_WIDTH = Math.min(900, window.innerWidth - 340);
  var drawerWidth = 440;

  function applyWidth(w) {
    drawerWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, w));
    document.documentElement.style.setProperty('--pdf-w', drawerWidth + 'px');
  }

  if (resizeHandle) {
    resizeHandle.addEventListener('mousedown', function(e) {
      e.preventDefault();
      resizeHandle.classList.add('dragging');
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
      var startX = e.clientX;
      var startW = overlay.offsetWidth;
      function onMove(e) { applyWidth(startW + (startX - e.clientX)); }
      function onUp() {
        resizeHandle.classList.remove('dragging');
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  window.addEventListener('resize', function() {
    var sidebarW = window.innerWidth <= 768 ? 0 : 280;
    MAX_WIDTH = Math.min(900, Math.max(320, window.innerWidth - sidebarW - 80));
    applyWidth(drawerWidth);
    var sidebar = document.getElementById('sidebar');
    if (sidebar && window.innerWidth <= 768 && overlay.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  });

  var currentPage = 1;
  var totalPages  = 1;
  var currentSrc  = null;
  var pdfDoc = null;
  var isRendering = false;
  var pageNumPending = null;

  // Initialize PDF.js
  if (window.pdfjsLib) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
  }

  function renderPage(num) {
    if (!pdfDoc || !canvas) return;
    if (isRendering) {
      pageNumPending = num;
      return;
    }
    isRendering = true;
    pdfDoc.getPage(num).then(function(page) {
      var ctx = canvas.getContext('2d');
      // Calculate scale to fit the canvas element's CSS width/height, but for quality we use a higher scale
      var scale = 1.5; 
      var viewport = page.getViewport({ scale: scale });
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      // Let it size naturally up to container width, like the old UI
      canvas.style.maxWidth = '100%';
      canvas.style.width = 'auto';
      canvas.style.height = 'auto';

      var pdfPageEl = document.getElementById('pdf-page-canvas');
      if (pdfPageEl) {
        pdfPageEl.style.width = '';
        pdfPageEl.style.maxWidth = '';
      }

      var renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };
      var renderTask = page.render(renderContext);
      renderTask.promise.then(function() {
        isRendering = false;
        
        // Apply highlighting if there's an excerpt
        if (currentSrc && currentSrc.excerpt) {
          highlightTextOnPdf(page, currentSrc.excerpt, viewport);
        }

        if (pageNumPending !== null) {
          renderPage(pageNumPending);
          pageNumPending = null;
        }
      });
    });
  }

  async function highlightTextOnPdf(page, searchText, viewport) {
    var existingOverlay = document.getElementById("pdf-highlight-overlay");
    if (existingOverlay) existingOverlay.remove();
    if (!searchText) return;

    try {
      var textContent = await page.getTextContent();
      var canvas = document.getElementById("pdf-render-canvas");
      if (!canvas) return;

      var overlay = document.createElement("div");
      overlay.id = "pdf-highlight-overlay";
      overlay.style.cssText = "position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10;";

      var searchNorm = searchText.toLowerCase().replace(/\s+/g, " ").trim();
      var searchWords = searchNorm.split(" ").filter(function(w) { return w.length > 4; }).slice(0, 15);
      if (searchWords.length === 0) return;

      var matchCount = 0;
      for (var i = 0; i < textContent.items.length; i++) {
        var item = textContent.items[i];
        if (!item.str || !item.str.trim()) continue;
        var itemText = item.str.toLowerCase();
        var hasMatch = searchWords.some(function(word) { return itemText.includes(word); });
        if (!hasMatch) continue;

        var tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
        var xPct = (tx[4] / viewport.width) * 100;
        var yPct = (tx[5] / viewport.height) * 100;
        var fontSizePct = (Math.sqrt(tx[2] * tx[2] + tx[3] * tx[3]) / viewport.height) * 100;
        var widthPct = ((item.width * viewport.scale) / viewport.width) * 100;
        var extraPaddingPct = (4 / viewport.height) * 100;

        var rect = document.createElement("div");
        rect.style.cssText = "position: absolute; left: " + xPct + "%; top: " + (yPct - fontSizePct) + "%; width: " + widthPct + "%; height: " + (fontSizePct + extraPaddingPct) + "%; background: rgba(16, 185, 129, 0.3); border-radius: 2px; pointer-events: none;";
        overlay.appendChild(rect);
        matchCount++;
      }

      if (matchCount > 0) {
        var wrapper = document.getElementById("pdf-canvas-wrapper");
        if (wrapper) wrapper.appendChild(overlay);
      }
    } catch (e) {
      console.error(e);
    }
  }

  async function openViewer(src) {
    currentSrc = src;
    currentPage = src.page || 1;
    
    var bookId = src.book_id || src.title;
    var token = localStorage.getItem("kr_token");
    
    titleEl.textContent = 'Loading...';
    subtitleEl.textContent = '';
    
    var layout = document.querySelector('.chat-layout');
    if (layout) layout.classList.add('pdf-open');
    overlay.classList.add('open');
    document.documentElement.style.setProperty('--pdf-w', drawerWidth + 'px');
    var sidebar = document.getElementById('sidebar');
    if (sidebar && window.innerWidth <= 768) sidebar.classList.remove('open');
    
    if (window.pdfjsLib && bookId) {
      const url = `${API}/pdf/${encodeURIComponent(bookId)}?token=${encodeURIComponent(token)}`;
      try {
        pdfDoc = await pdfjsLib.getDocument(url).promise;
        totalPages = pdfDoc.numPages;
        currentPage = Math.max(1, Math.min(currentPage, totalPages));
      } catch (err) {
        console.error("Failed to load PDF:", err);
      }
    }
    
    titleEl.textContent = src.title || bookId || 'PDF Document';
    subtitleEl.textContent = 'p.' + currentPage + ' · PDF Document';
    
    updatePageControls();
    if (pdfDoc) {
      renderPage(currentPage);
    }
  }

  function closeViewer() {
    overlay.classList.remove('open');
    var layout = document.querySelector('.chat-layout');
    if (layout) setTimeout(function() { layout.classList.remove('pdf-open'); }, 300);
  }

  closeBtn.addEventListener('click', closeViewer);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && overlay.classList.contains('open')) closeViewer();
  });

  function updatePageControls() {
    curPageEl.textContent = currentPage;
    totPageEl.textContent = totalPages;
    badgeEl.textContent   = 'p. ' + currentPage;
    prevBtn.disabled = currentPage <= 1;
    nextBtn.disabled = currentPage >= totalPages;
  }

  prevBtn.addEventListener('click', function() {
    if (currentPage > 1) {
      currentPage--;
      updatePageControls();
      renderPage(currentPage);
    }
  });

  nextBtn.addEventListener('click', function() {
    if (currentPage < totalPages) {
      currentPage++;
      updatePageControls();
      renderPage(currentPage);
    }
  });

  document.addEventListener('click', function(e) {
    // Source card click
    var card = e.target.closest('.source-card');
    if (card) {
      var title   = card.getAttribute('data-src-title');
      var book_id = card.getAttribute('data-src-book_id') || title;
      var page    = parseInt(card.getAttribute('data-src-page'), 10) || 1;
      var excerpt = card.getAttribute('data-src-excerpt') || '';
      if (title || book_id) openViewer({ title: title, book_id: book_id, page: page, excerpt: excerpt });
      return;
    }

    // Inline citation chip click
    var chip = e.target.closest('.inline-citation-chip');
    if (chip) {
      var bookName = chip.getAttribute('data-cite-book') || '';
      var page     = parseInt(chip.getAttribute('data-cite-page'), 10) || 1;
      var section  = chip.getAttribute('data-cite-section') || '';

      // The book field in citations is often the book_id itself (e.g. "gemini-prompt").
      // 1. Start with the raw book name as the book_id
      var bookId = bookName;

      // 2. Try to match against booksMap display names (e.g. "Clean Code" -> "clean-code")
      Object.keys(booksMap).forEach(function(id) {
        if (booksMap[id].name.toLowerCase() === bookName.toLowerCase()) {
          bookId = id;
        }
      });

      // 3. Only fall back to the active session's book if the bookName is completely empty
      if (!bookId) {
        var activeSession = state.sessionMap[state.activeSessionId];
        if (activeSession) bookId = activeSession.book;
      }

      openViewer({ title: bookName, book_id: bookId, page: page, excerpt: section });
    }
  });
}
