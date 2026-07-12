(function () {
  'use strict';

  const API = "";

  function showError(el, msg) {
    el.textContent = msg;
    el.style.display = 'block';
  }

  function clearError(el) {
    el.textContent = '';
    el.style.display = 'none';
  }

  /* ===== LOGIN ===== */
  var loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      var errorEl = document.getElementById('login-error');
      clearError(errorEl);

      var email = document.getElementById('login-email').value.trim();
      var password = document.getElementById('login-password').value;
      const submitBtn = loginForm.querySelector('button[type="submit"]');

      if (!email || !password) {
        showError(errorEl, 'Please fill in all fields.');
        return;
      }

      try {
        if(submitBtn) submitBtn.disabled = true;
        const res = await fetch(API + "/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) {
            showError(errorEl, data.detail || "Login failed.");
            return;
        }
        
        localStorage.setItem("krutrim-auth", "true");
        localStorage.setItem("kr_token", data.token);
        localStorage.setItem("kr_user", JSON.stringify({
            user_id: data.user_id,
            name: data.name,
            email: data.email,
            role: data.role,
        }));
        
        window.location.href = '/';
      } catch(err) {
        showError(errorEl, "Cannot connect to server.");
      } finally {
        if(submitBtn) submitBtn.disabled = false;
      }
    });
  }

  /* ===== SIGNUP ===== */
  var signupForm = document.getElementById('signup-form');
  if (signupForm) {
    signupForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      var errorEl = document.getElementById('signup-error');
      clearError(errorEl);

      var name = document.getElementById('signup-name').value.trim();
      var email = document.getElementById('signup-email').value.trim();
      var password = document.getElementById('signup-password').value;
      var confirm = document.getElementById('signup-confirm').value;
      const submitBtn = signupForm.querySelector('button[type="submit"]');

      if (!name || !email || !password || !confirm) {
        showError(errorEl, 'Please fill in all fields.');
        return;
      }

      if (password.length < 8) {
        showError(errorEl, 'Password must be at least 8 characters.');
        return;
      }

      if (password !== confirm) {
        showError(errorEl, 'Passwords do not match.');
        return;
      }

      try {
        if(submitBtn) submitBtn.disabled = true;
        const res = await fetch(API + "/auth/signup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email, password }),
        });
        const data = await res.json();
        if (!res.ok) {
            showError(errorEl, data.detail || "Signup failed.");
            return;
        }
        
        // Auto-login or redirect to login (the old code set email in login form and showed it)
        // Here we just login directly
        const loginRes = await fetch(API + "/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const loginData = await loginRes.json();
        if(loginRes.ok) {
            localStorage.setItem("krutrim-auth", "true");
            localStorage.setItem("kr_token", loginData.token);
            localStorage.setItem("kr_user", JSON.stringify({
                user_id: loginData.user_id,
                name: loginData.name,
                email: loginData.email,
                role: loginData.role,
            }));
            window.location.href = '/';
        } else {
            window.location.href = '/pages/login.html';
        }
      } catch(err) {
        showError(errorEl, "Cannot connect to server.");
      } finally {
        if(submitBtn) submitBtn.disabled = false;
      }
    });
  }

  /* ===== THEME TOGGLE ===== */
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
})();
