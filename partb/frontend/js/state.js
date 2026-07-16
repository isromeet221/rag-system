export const API = "";

export const state = {
  activeSessionId: null,
  sessionMap: {},
  nextId: 100, // Not really needed if API provides IDs
};

export const booksMap = {};

export async function loadBooks() {
  try {
    const res = await apiFetch('/library');
    if (!res) return;
    const data = await res.json();
    const books = data.books || [];
    for (const b of books) {
      booksMap[b.book_id] = {
        name: b.title || b.book_id,
        pages: b.total_pages || 0,
        chunks: b.total_chunks || 0
      };
    }
    return books;
  } catch (e) {
    console.error("Failed to load books", e);
    return [];
  }
}

export function authHeaders() {
  const token = localStorage.getItem("kr_token");
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
}

export async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, {
    ...opts,
    headers: { ...authHeaders(), ...(opts.headers || {}) },
  });
  if (res.status === 401) {
    localStorage.removeItem("krutrim-auth");
    localStorage.removeItem("kr_token");
    localStorage.removeItem("kr_user");
    window.location.href = '/pages/login.html';
    return null;
  }
  return res;
}

export async function loadChats() {
  try {
    const res = await apiFetch('/chats');
    if (!res) return;
    const data = await res.json();
    state.sessionMap = {};
    if (data && data.length > 0) {
      data.forEach(chat => {
        const id = chat.chat_id || chat.id || chat._id;
        if (!id) return;
        state.sessionMap[id] = {
          id: id,
          name: chat.title || 'Untitled',
          book: (chat.book_ids && chat.book_ids[0]) || chat.book_id || '',
          model: chat.default_mode || chat.model || 'balanced',
          messages: [] // loaded on demand
        };
      });
      // Optionally sort by created_at if available
    }
    return Object.keys(state.sessionMap);
  } catch (e) {
    console.error("Failed to load chats", e);
    return [];
  }
}

export async function loadMessages(chatId) {
  if (!chatId) return [];
  try {
    const res = await apiFetch(`/chats/${chatId}/messages`);
    if (!res) return [];
    const msgs = await res.json();
    if (state.sessionMap[chatId]) {
      state.sessionMap[chatId].messages = msgs.map(m => ({
        sender: m.role === 'user' ? 'user' : 'bot',
        text: m.content,
        sources: m.sources || []
      }));
    }
    return msgs;
  } catch (e) {
    console.error("Failed to load messages", e);
    return [];
  }
}

export function escapeHtml(str) {
  var div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

export function truncate(str, len) {
  if (str.length <= len) return str;
  return str.substring(0, len) + '...';
}
