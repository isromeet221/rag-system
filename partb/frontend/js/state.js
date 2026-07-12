export const API = "";

export const state = {
  activeSessionId: null,
  sessionMap: {},
  nextId: 100, // Not really needed if API provides IDs
};

export const booksMap = {
  'clean-code': { name: 'Clean Code', pages: 464, chunks: 1856 },
  'pragmatic-programmer': { name: 'The Pragmatic Programmer', pages: 352, chunks: 1408 },
  'sicp': { name: 'SICP', pages: 657, chunks: 2628 },
  'clrs': { name: 'CLRS Algorithms', pages: 1312, chunks: 5248 },
  'ydkjs': { name: "You Don't Know JS", pages: 250, chunks: 1000 }
};

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
          book: (chat.book_ids && chat.book_ids[0]) || chat.book_id || 'clean-code',
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
