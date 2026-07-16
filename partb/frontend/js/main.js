import { state, booksMap, API, apiFetch, authHeaders, loadChats, loadMessages, loadBooks } from './state.js';
import {
  el, renderSessions, renderMessages, updateHeader,
  showTypingIndicator, removeTypingIndicator,
  openNewChatModal, closeNewChatModal,
  openDeleteContextModal, closeDeleteContextModal,
  openSettingsModal, closeSettingsModal,
  openSearchModal, closeSearchModal,
  initCustomFormDropdown, populateBookDropdowns
} from './render.js?v=32';
import { initPdfViewer } from './pdf.js?v=32';

let currentAbortController = null;
const stopGenerateBtn = document.getElementById('stop-generate-btn');
const sendBtn = document.getElementById('send-message-btn');

if (stopGenerateBtn) {
  stopGenerateBtn.addEventListener('click', () => {
    if (currentAbortController) {
      currentAbortController.abort();
    }
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// In-place update of the last bot message's content — avoids full DOM rebuild flicker
function updateLastBotContent(text) {
  if (!el.messageThread) return;
  var msgs = el.messageThread.querySelectorAll('.message.bot');
  if (msgs.length === 0) return;
  var last = msgs[msgs.length - 1];
  var contentDiv = last.querySelector('.message-content');
  if (!contentDiv) return;

  // Strip citation patterns before markdown parsing
  var cleanText = text.replace(/\[Book:[^\]]*?\]/g, '');

  if (window.marked) {
    contentDiv.innerHTML = marked.parse(cleanText, { breaks: true });
  } else {
    contentDiv.textContent = text;
  }

  // Auto-scroll to bottom
  if (el.chatMessages) {
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }
}

if (!localStorage.getItem('krutrim-auth')) {
  window.location.href = '/pages/login.html';
}

async function switchSession(id) {
  if (id === state.activeSessionId) return;
  state.activeSessionId = id;
  renderSessions();
  updateHeader();
  if (el.messageThread) el.messageThread.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--mute);">Loading...</div>';
  await loadMessages(id);
  renderMessages();
  el.chatInput.disabled = false;
  el.chatInput.focus();
}

async function sendMessage(text) {
  var trimmed = text.trim();
  if (!trimmed) return;
  if (!state.activeSessionId || !state.sessionMap[state.activeSessionId]) {
    await createNewSession("New Chat", "clean-code", "balanced");
  }
  var s = state.sessionMap[state.activeSessionId];
  s.messages.push({ sender: 'user', text: trimmed });
  renderMessages();
  el.chatForm.reset();
  el.chatInput.style.height = 'auto';
  el.chatInput.disabled = true;
  showTypingIndicator();
  
  if (sendBtn && stopGenerateBtn) {
    sendBtn.style.display = 'none';
    stopGenerateBtn.style.display = 'inline-flex';
  }
  
  currentAbortController = new AbortController();

  try {
    const res = await apiFetch(`/chats/${state.activeSessionId}/ask`, {
      method: "POST",
      body: JSON.stringify({ question: trimmed, mode: s.model || "balanced" }),
      signal: currentAbortController.signal
    });
    
    if (!res || !res.ok) throw new Error("Failed to send message");
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fulltext = "";
    let sources = [];
    let buf = "";

    removeTypingIndicator();
    s.messages.push({ sender: 'bot', text: "Thinking...", isStreaming: true });
    renderMessages();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch(e) { continue; }
        
        if (evt.type === "status") {
            s.messages[s.messages.length - 1].text = `Thinking: ${evt.message}...`;
            updateLastBotContent(s.messages[s.messages.length - 1].text);
        } else if (evt.type === "token") {
          fulltext += evt.content;
          s.messages[s.messages.length - 1].text = fulltext;
          updateLastBotContent(fulltext);
        } else if (evt.type === "done") {
          sources = evt.sources || [];
          s.messages[s.messages.length - 1].sources = sources;
          s.messages[s.messages.length - 1].text = fulltext;
          delete s.messages[s.messages.length - 1].isStreaming;
          renderMessages();
        } else if (evt.type === "error") {
          s.messages[s.messages.length - 1].text = `Error: ${evt.message}`;
          delete s.messages[s.messages.length - 1].isStreaming;
          renderMessages();
        }
      }
    }
  } catch(e) {
    console.error(e);
    removeTypingIndicator();
    if (e.name === 'AbortError') {
      s.messages[s.messages.length - 1].text += " (Stopped)";
      delete s.messages[s.messages.length - 1].isStreaming;
    } else {
      s.messages.push({ sender: 'bot', text: "Error: Could not get response." });
    }
    renderMessages();
  } finally {
    if (sendBtn && stopGenerateBtn) {
      stopGenerateBtn.style.display = 'none';
      sendBtn.style.display = 'inline-flex';
    }
    el.chatInput.disabled = false;
    el.chatInput.focus();
    currentAbortController = null;
  }
}

async function createNewSession(title, book, model) {
  try {
    var fallbackBook = book || Object.keys(booksMap)[0] || null;
    if (!fallbackBook) {
      alert('No books available. Please upload a book first.');
      return;
    }
    const res = await apiFetch("/chats", {
      method: "POST",
      body: JSON.stringify({ book_ids: [fallbackBook], default_mode: model || "balanced", title: title || "New Chat" })
    });
    if(!res || !res.ok) return;
    const data = await res.json();
    const id = data.id || data.chat_id || data._id; // fallback for different backends
    if (id) {
      state.sessionMap[id] = {
        id: id,
        name: data.title || title,
        book: (data.book_ids && data.book_ids[0]) || book,
        model: data.default_mode || model,
        messages: []
      };
      state.activeSessionId = id;
      renderSessions();
      updateHeader();
      renderMessages();
      el.chatInput.disabled = false;
      el.chatInput.focus();
    }
  } catch(e) {
    console.error("Failed to create session", e);
  }
}

// -- Event listeners --

el.chatForm.addEventListener('submit', function (e) {
  e.preventDefault();
  var text = el.chatInput.value;
  if (text.trim()) sendMessage(text);
});

if (el.chatInput) {
  el.chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      var text = el.chatInput.value;
      if (text.trim()) sendMessage(text);
    }
  });
}

if (el.newSessionBtn) {
  el.newSessionBtn.addEventListener('click', function () { openNewChatModal(); });
}

var toggleBtn = document.getElementById('sidebar-toggle');
if (toggleBtn) {
  toggleBtn.addEventListener('click', function () { 
    if (window.innerWidth > 768) {
      document.querySelector('.chat-layout').classList.toggle('sidebar-closed');
    } else {
      el.sidebar.classList.toggle('open'); 
    }
  });
  document.addEventListener('click', function (e) {
    if (window.innerWidth <= 768 && el.sidebar.classList.contains('open') &&
        !el.sidebar.contains(e.target) && e.target !== toggleBtn && !toggleBtn.contains(e.target)) {
      el.sidebar.classList.remove('open');
    }
  });
}

// Session search
var searchInput = document.getElementById('session-search');
if (searchInput) {
  searchInput.addEventListener('input', function () {
    var q = this.value.toLowerCase().trim();
    var items = document.querySelectorAll('.chat-session');
    var visibleCount = 0;
    items.forEach(function (item) {
      var name = item.querySelector('.chat-session-name');
      var match = !q || (name && name.textContent.toLowerCase().includes(q));
      item.style.display = match ? '' : 'none';
      if (match) visibleCount++;
    });
    var emptyMsg = document.querySelector('.sidebar-list-empty');
    if (items.length > 0 && visibleCount === 0) {
      if (!emptyMsg) {
        emptyMsg = document.createElement('div');
        emptyMsg.className = 'sidebar-list-empty';
        emptyMsg.textContent = 'No chats found';
        document.getElementById('session-list').appendChild(emptyMsg);
      }
    } else if (emptyMsg) {
      emptyMsg.remove();
    }
  });
}

// Scroll to bottom button
var scrollBtn = document.getElementById('scroll-to-bottom-btn');
var messagesContainer = document.getElementById('chat-messages');
if (scrollBtn && messagesContainer) {
  messagesContainer.addEventListener('scroll', function() {
    var distanceToBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
    if (distanceToBottom > 150) {
      scrollBtn.classList.add('visible');
    } else {
      scrollBtn.classList.remove('visible');
    }
  });
  
  scrollBtn.addEventListener('click', function() {
    messagesContainer.scrollTo({
      top: messagesContainer.scrollHeight,
      behavior: 'smooth'
    });
  });
}

var sidebarOpenSearchBtn = document.getElementById('sidebar-search-trigger');
if (sidebarOpenSearchBtn) {
  sidebarOpenSearchBtn.addEventListener('click', function() {
    openSearchModal();
  });
}

// Theme toggle
var themeToggle = document.getElementById('theme-toggle');
var savedTheme = localStorage.getItem('krutrim-theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
if (themeToggle) {
  themeToggle.innerHTML = savedTheme === 'dark'
    ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  themeToggle.addEventListener('click', function () {
    var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('krutrim-theme', next);
    themeToggle.innerHTML = next === 'dark'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  });
}

// Model dropdown
var modelDropdown = document.getElementById('model-dropdown');
var modelToggle = document.getElementById('model-dropdown-toggle');
if (modelDropdown && modelToggle) {
  modelToggle.addEventListener('click', function(e) {
    e.stopPropagation();
    modelDropdown.classList.toggle('open');
  });
  document.addEventListener('click', function(e) {
    if (!modelDropdown.contains(e.target)) modelDropdown.classList.remove('open');
  });
  document.querySelectorAll('#model-dropdown-menu .dropdown-item').forEach(function(item) {
    item.addEventListener('click', function() {
      var items = document.querySelectorAll('#model-dropdown-menu .dropdown-item');
      items.forEach(function(i) { i.classList.remove('selected'); });
      item.classList.add('selected');
      document.getElementById('model-dropdown-label').textContent = item.querySelector('span').textContent;
      var svg = item.querySelector('svg').cloneNode(true);
      var oldSvg = modelToggle.querySelector('.dropdown-icon');
      if (oldSvg) { svg.classList.add('dropdown-icon'); modelToggle.replaceChild(svg, oldSvg); }
      modelDropdown.classList.remove('open');
      var modelVal = item.getAttribute('data-value');
      if (state.activeSessionId && state.sessionMap[state.activeSessionId]) {
        state.sessionMap[state.activeSessionId].model = modelVal;
      }
    });
  });
}

// Session click delegation
document.addEventListener('click', function(e) {
  var sessionDiv = e.target.closest('.chat-session');
  if (sessionDiv && !e.target.closest('.chat-session-actions')) {
    switchSession(sessionDiv.dataset.id);
  }
});

// Custom event for switching sessions
document.addEventListener('switch-session', function(e) {
  if (e.detail && e.detail.id) {
    switchSession(e.detail.id);
  }
});

// Rename / Delete delegation
document.addEventListener('click', function(e) {
  var renameBtn = e.target.closest('.rename-btn');
  var deleteBtn = e.target.closest('.delete-btn');
  if (!renameBtn && !deleteBtn) return;
  var sessionDiv = e.target.closest('.chat-session');
  if (!sessionDiv) return;
  var id = sessionDiv.dataset.id;
  var s = state.sessionMap[id];
  if (!s) return;

  if (renameBtn) {
    var newName = prompt('Enter new chat name:', s.name);
    if (newName !== null && newName.trim() !== '') {
      s.name = newName.trim();
      apiFetch(`/chats/${id}`, { method: "PATCH", body: JSON.stringify({ title: s.name }) });
      renderSessions();
      updateHeader();
    }
  }

  if (deleteBtn) {
    if (confirm('Are you sure you want to delete this chat?')) {
      delete state.sessionMap[id];
      apiFetch(`/chats/${id}`, { method: "DELETE" });
      if (state.activeSessionId === id) {
        var remainingIds = Object.keys(state.sessionMap);
        state.activeSessionId = remainingIds.length > 0 ? remainingIds[0] : null;
        if (state.activeSessionId) {
            switchSession(state.activeSessionId);
        } else {
            renderSessions();
            updateHeader();
            renderMessages();
            el.chatInput.disabled = true;
        }
      } else {
        renderSessions();
      }
    }
  }
});

document.addEventListener('click', function(e) {
  if (!e.target.closest('.chat-session-actions')) {
    document.querySelectorAll('.chat-session-actions.open').forEach(function(el2) { el2.classList.remove('open'); });
  }
});

// Auto-expand textarea
if (el.chatInput) {
  el.chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
  });
}

// New chat modal
if (el.newChatForm) {
  el.newChatForm.addEventListener('submit', function (e) {
    e.preventDefault();
    var title = el.modalChatTitle.value.trim();
    var book = el.modalBookSelect.value;
    var modelInput = el.newChatForm.querySelector('input[name="modal-model"]:checked');
    var model = modelInput ? modelInput.value : 'fast';
    if (title && book) { createNewSession(title, book, model); closeNewChatModal(); }
  });
}

if (el.modalCloseBtn) el.modalCloseBtn.addEventListener('click', closeNewChatModal);
if (el.modalCancelBtn) el.modalCancelBtn.addEventListener('click', closeNewChatModal);
if (el.newChatModal) {
  el.newChatModal.addEventListener('click', function (e) { if (e.target === el.newChatModal) closeNewChatModal(); });
}
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && el.newChatModal && el.newChatModal.classList.contains('open')) closeNewChatModal();
});

// Delete context modal
var openDeleteContextBtn = document.getElementById('open-delete-context-btn');
var deleteContextModal = document.getElementById('delete-context-modal');
var deleteContextForm = document.getElementById('delete-context-form');

if (openDeleteContextBtn) {
  openDeleteContextBtn.addEventListener('click', function() {
    closeSettingsModal();
    openDeleteContextModal();
  });
}
if (document.getElementById('delete-context-close')) document.getElementById('delete-context-close').addEventListener('click', closeDeleteContextModal);
if (document.getElementById('delete-context-cancel')) document.getElementById('delete-context-cancel').addEventListener('click', closeDeleteContextModal);

if (deleteContextForm) {
  deleteContextForm.addEventListener('submit', function(e) {
    e.preventDefault();
    var select = document.getElementById('delete-book-select');
    var bookId = select ? select.value : null;
    if (bookId) {
      var bookObj = booksMap[bookId];
      alert('Context for "' + (bookObj ? bookObj.name : bookId) + '" has been successfully deleted from the database.');
      closeDeleteContextModal();
    }
  });
}

if (deleteContextModal) {
  deleteContextModal.addEventListener('click', function(e) { if (e.target === deleteContextModal) closeDeleteContextModal(); });
}

// Settings modal
var settingsModal = document.getElementById('settings-modal');
if (document.getElementById('settings-modal-close')) {
  document.getElementById('settings-modal-close').addEventListener('click', closeSettingsModal);
}
if (settingsModal) {
  settingsModal.addEventListener('click', function(e) { if (e.target === settingsModal) closeSettingsModal(); });
}
var settingsDarkCb = document.getElementById('settings-dark-mode');
if (settingsDarkCb) {
  settingsDarkCb.addEventListener('change', function() {
    var next = this.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('krutrim-theme', next);
    var tt = document.getElementById('theme-toggle');
    if (tt) {
      tt.innerHTML = next === 'dark'
        ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
    }
  });
}
if (document.getElementById('user-settings-btn')) {
  document.getElementById('user-settings-btn').addEventListener('click', openSettingsModal);
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    if (deleteContextModal && deleteContextModal.classList.contains('open')) closeDeleteContextModal();
    if (settingsModal && settingsModal.classList.contains('open')) closeSettingsModal();
    if (el.searchModal && el.searchModal.classList.contains('open')) {
      if (el.searchInput && el.searchInput.value.trim().length > 0) {
        el.searchInput.value = '';
        el.searchInput.dispatchEvent(new Event('input'));
      } else {
        closeSearchModal();
      }
    }
  }
  // Search modal shortcut (Ctrl+K or Cmd+K)
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    if (el.searchModal && el.searchModal.classList.contains('open')) {
      closeSearchModal();
    } else {
      openSearchModal();
    }
  }
});

// Search functionality
if (el.searchInput) {
  el.searchInput.addEventListener('input', function() {
    var q = this.value.toLowerCase().trim();
    if (!el.searchResults) return;
    
    el.searchResults.innerHTML = '';
    var s = state.sessionMap[state.activeSessionId];
    
    if (!q) {
      // 1. Add "New chat" button
      var newChatDiv = document.createElement('div');
      newChatDiv.style.cssText = 'padding: 12px var(--space-lg); cursor: pointer; display: flex; align-items: center; gap: 12px; margin-bottom: 8px; border-radius: var(--radius-sm); margin-left: 8px; margin-right: 8px;';
      newChatDiv.onmouseover = function() { newChatDiv.style.backgroundColor = 'var(--surface-soft)'; };
      newChatDiv.onmouseout = function() { newChatDiv.style.backgroundColor = 'transparent'; };
      
      newChatDiv.innerHTML = 
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--ink);"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>' +
        '<div style="font-size: 15px; color: var(--ink); font-weight: var(--weight-regular);">New chat</div>';
        
      newChatDiv.addEventListener('click', function() {
        closeSearchModal();
        openNewChatModal();
      });
      el.searchResults.appendChild(newChatDiv);

      var sessionIds = Object.keys(state.sessionMap);
      if (sessionIds.length > 0) {
        // Render chat items
        sessionIds.forEach(function(id) {
          var session = state.sessionMap[id];
          var div = document.createElement('div');
          div.style.cssText = 'padding: 12px var(--space-lg); cursor: pointer; display: flex; align-items: center; gap: 12px; border-radius: var(--radius-sm); margin-left: 8px; margin-right: 8px;';
          div.onmouseover = function() { div.style.backgroundColor = 'var(--surface-soft)'; };
          div.onmouseout = function() { div.style.backgroundColor = 'transparent'; };
          
          div.innerHTML = 
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--ink);"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>' +
            '<div style="font-size: 15px; color: var(--ink); font-weight: var(--weight-regular);">' + escapeHtml(session.name) + '</div>';
            
          div.addEventListener('click', function() {
            var evt = new CustomEvent('switch-session', { detail: { id: id } });
            document.dispatchEvent(evt);
            closeSearchModal();
          });
          el.searchResults.appendChild(div);
        });
      }
      return;
    }
    
    if (!s && sessionIds.length === 0) {
      if (el.searchResultsEmpty) {
        el.searchResults.appendChild(el.searchResultsEmpty);
        el.searchResultsEmpty.style.display = 'block';
        el.searchResultsEmpty.textContent = 'No chats to search.';
      }
      return;
    }

    var matches = 0;
    
    // 1. Search Chat Titles
    var sessionIds = Object.keys(state.sessionMap);
    sessionIds.forEach(function(id) {
      var session = state.sessionMap[id];
      if (session.name.toLowerCase().includes(q)) {
        matches++;
        var div = document.createElement('div');
        div.style.cssText = 'padding: 12px var(--space-lg); cursor: pointer; display: flex; align-items: center; gap: 12px; border-radius: var(--radius-sm); margin-left: 8px; margin-right: 8px;';
        div.onmouseover = function() { div.style.backgroundColor = 'var(--surface-soft)'; };
        div.onmouseout = function() { div.style.backgroundColor = 'transparent'; };
        
        var escapedName = escapeHtml(session.name);
        var regex = new RegExp('(' + q.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&') + ')', 'gi');
        var highlightedName = escapedName.replace(regex, '<mark style="background: rgba(255, 215, 0, 0.4); color: inherit; border-radius: 2px; padding: 0 2px;">$1</mark>');

        div.innerHTML = 
          '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--ink); flex-shrink: 0;"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>' +
          '<div style="font-size: 15px; color: var(--ink); font-weight: var(--weight-regular);">' + highlightedName + '</div>';
          
        div.addEventListener('click', function() {
          var evt = new CustomEvent('switch-session', { detail: { id: id } });
          document.dispatchEvent(evt);
          closeSearchModal();
        });
        el.searchResults.appendChild(div);
      }
    });

    // 2. Search Messages
    if (s && s.messages) {
      s.messages.forEach(function(msg, idx) {
        if (msg.text.toLowerCase().includes(q)) {
          matches++;
          var div = document.createElement('div');
          div.style.cssText = 'padding: 12px var(--space-lg); cursor: pointer; border-radius: var(--radius-sm); margin-left: 8px; margin-right: 8px; display: flex; flex-direction: column; gap: 4px;';
          div.onmouseover = function() { div.style.backgroundColor = 'var(--surface-soft)'; };
          div.onmouseout = function() { div.style.backgroundColor = 'transparent'; };
          
          var senderName = msg.sender === 'user' ? 'User' : 'Krutrim AI';
          
          var textSnippet = "";
          var lowerMsg = msg.text.toLowerCase();
          var matchIdx = lowerMsg.indexOf(q);
          if (matchIdx !== -1) {
            var start = Math.max(0, matchIdx - 40);
            var end = Math.min(msg.text.length, matchIdx + q.length + 80);
            textSnippet = msg.text.substring(start, end);
            if (start > 0) textSnippet = '...' + textSnippet;
            if (end < msg.text.length) textSnippet = textSnippet + '...';
          } else {
            textSnippet = msg.text.substring(0, 150) + (msg.text.length > 150 ? '...' : '');
          }
          
          // Highlight the match
          var escapedSnippet = escapeHtml(textSnippet);
          var regex = new RegExp('(' + q.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&') + ')', 'gi');
          var highlighted = escapedSnippet.replace(regex, '<mark style="background: rgba(255, 215, 0, 0.4); color: inherit; border-radius: 2px; padding: 0 2px;">$1</mark>');
          
          div.innerHTML = 
            '<div style="font-size: 12px; color: var(--mute); font-weight: var(--weight-medium); text-transform: uppercase; letter-spacing: 0.05em;">' + senderName + '</div>' +
            '<div style="font-size: 15px; color: var(--ink); line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis;">' + highlighted + '</div>';
            
          div.addEventListener('click', function() {
            closeSearchModal();
            // Ideally we would scroll to the message, but for now we just close it
          });
          el.searchResults.appendChild(div);
        }
      });
    }

    if (matches === 0) {
      if (el.searchResultsEmpty) {
        el.searchResults.appendChild(el.searchResultsEmpty);
        el.searchResultsEmpty.style.display = 'block';
        el.searchResultsEmpty.textContent = 'No matches found.';
      }
    }
  });
}

var searchModalCloseBtn = document.getElementById('search-modal-close');
if (searchModalCloseBtn) {
  searchModalCloseBtn.addEventListener('click', closeSearchModal);
}
if (el.searchModal) {
  el.searchModal.addEventListener('click', function(e) { if (e.target === el.searchModal) closeSearchModal(); });
}

// Init — initialize dropdowns and PDF viewer at module level
initCustomFormDropdown('modal-book-dropdown', 'modal-book-toggle', 'modal-book-label', 'modal-book-select', 'modal-book-menu');
initCustomFormDropdown('delete-book-dropdown', 'delete-book-toggle', 'delete-book-label', 'delete-book-select', 'delete-book-menu');
initPdfViewer();

async function init() {
  // Load real books from API
  const books = await loadBooks();
  if (books && books.length > 0) {
    populateBookDropdowns(books);
  }
  
  let currentUser = null;
  try {
    currentUser = JSON.parse(localStorage.getItem('kr_user'));
  } catch (e) {}
  if (currentUser) {
    var nameEl = document.getElementById('sidebar-user-name');
    var roleEl = document.getElementById('sidebar-user-role');
    if (nameEl && currentUser.name) nameEl.textContent = currentUser.name;
    if (roleEl && currentUser.role) roleEl.textContent = currentUser.role;
  }

  const ids = await loadChats();
  if (ids && ids.length > 0) {
    await switchSession(ids[0]);
  } else {
    updateHeader();
    renderSessions();
  }
}
init();
