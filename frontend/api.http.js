/* Real API client for the Ergasia UI.
 * Same method names and return shapes as the mock `api` inside index.html, so the UI is unchanged.
 * Loaded before the main script. Add ?mock=1 to the URL to fall back to the built-in mock data.
 * Backend URL: same origin under /api by default. Override with window.ERGASIA_API_BASE.
 */
(function () {
  if (new URLSearchParams(location.search).has('mock')) return;
  const BASE = window.ERGASIA_API_BASE || '/api';
  const TK = 'ergasia.token';
  const getTok = () => { try { return localStorage.getItem(TK); } catch (_) { return null; } };
  const setTok = t => { try { t ? localStorage.setItem(TK, t) : localStorage.removeItem(TK); } catch (_) {} };

  async function req(method, path, body) {
    const headers = {}; let payload;
    const t = getTok(); if (t) headers.Authorization = 'Bearer ' + t;
    if (body instanceof FormData) payload = body;
    else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
    let r;
    try { r = await fetch(BASE + path, { method, headers, body: payload }); }
    catch (_) { throw new Error('Cannot reach the server. Check that the backend is running.'); }
    if (!r.ok) {
      let msg = 'Request failed (' + r.status + ').';
      try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : (j.detail && j.detail[0] && j.detail[0].msg) || msg; } catch (_) {}
      if (r.status === 401 && path !== '/auth/login' && path !== '/me') setTok(null);
      throw new Error(msg);
    }
    return r.status === 204 ? null : r.json();
  }
  const get = p => req('GET', p), post = (p, b) => req('POST', p, b === undefined ? {} : b), patch = (p, b) => req('PATCH', p, b);

  window.ERGASIA_API = {
    /* auth */
    async login(role, email, password) { const r = await post('/auth/login', { email, password, role }); setTok(r.token); return r; },
    logout() { setTok(null); },
    me: () => get('/me'),
    /* student */
    studentAssignments: () => get('/student/assignments'),
    variant: vid => get('/student/assignments/' + vid),
    async submit({ variant_id, answers, files }) {
      const fd = new FormData(); const withFile = (files || []).filter(f => f.file);
      fd.append('payload', JSON.stringify({ variant_id, answers, files: withFile.map(f => ({ question_id: f.question_id, name: f.name })) }));
      withFile.forEach(f => fd.append('files', f.file, f.name));
      return req('POST', '/student/submissions', fd);
    },
    submission: id => get('/student/submissions/' + id),
    requestRegrade: (id, reason) => post('/student/submissions/' + id + '/regrade', { reason }),
    progress: () => get('/student/progress'),
    /* teacher */
    course: () => get('/teacher/course'),
    teacherDash: () => get('/teacher/dashboard'),
    createAssignment: f => post('/teacher/assignments', f),
    assignment: id => get('/teacher/assignments/' + id),
    generate: id => post('/teacher/assignments/' + id + '/generate'),
    async job(jid) { const j = await get('/jobs/' + jid); if (j.status === 'failed') throw new Error(j.error || 'Generation failed.'); return j; },
    variants: aid => get('/teacher/assignments/' + aid + '/variants'),
    patchVariant: (vid, body) => patch('/teacher/variants/' + vid, body),
    regenerate: vid => post('/teacher/variants/' + vid + '/regenerate'),
    release: aid => post('/teacher/assignments/' + aid + '/release'),
    gradingQueue: () => get('/teacher/grading-queue'),
    patchGrade: (id, body) => patch('/teacher/grades/' + id, body),
    analyticsMastery: () => get('/teacher/analytics/mastery'),
    analyticsCoPo: () => get('/teacher/analytics/co-po'),
    kbList: () => get('/teacher/knowledge-base'),
    kbUpload(file) { const fd = new FormData(); fd.append('file', file, file.name); return req('POST', '/teacher/knowledge-base', fd); },
  };
})();
