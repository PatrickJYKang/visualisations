document.addEventListener('DOMContentLoaded', () => {
  const teamA = document.getElementById('teamA');
  const teamB = document.getElementById('teamB');
  const listA = document.getElementById('listA');
  const listB = document.getElementById('listB');
  const form = document.getElementById('genForm');
  const submitBtn = document.getElementById('genBtn');
  const spinner = document.getElementById('spinner');
  const toast = document.getElementById('toast');
  const results = document.getElementById('results');
  const cardEl = document.querySelector('.card');

  async function fetchSuggestions(q) {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error('Search failed');
    return res.json();
  }

  let searchAbortA = null;
  let searchAbortB = null;

  async function handleSuggest(inputEl, listEl, abortRef) {
    const q = inputEl.value.trim();
    listEl.innerHTML = '';
    if (q.length < 2) return;
    try {
      if (abortRef && abortRef.abortController) abortRef.abortController.abort();
      const controller = new AbortController();
      abortRef.abortController = controller;
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`, { signal: controller.signal });
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      listEl.innerHTML = '';
      for (const s of data) {
        const opt = document.createElement('option');
        opt.value = s.name;
        opt.setAttribute('data-slug', s.file_slug || '');
        listEl.appendChild(opt);
      }
    } catch (e) {
      // ignore aborts
    }
  }

  teamA.addEventListener('input', () => handleSuggest(teamA, listA, (searchAbortA ||= {})));
  teamB.addEventListener('input', () => handleSuggest(teamB, listB, (searchAbortB ||= {})));

  function showToast(msg, kind = 'error') {
    toast.textContent = msg;
    toast.style.display = 'block';
    toast.style.background = kind === 'error' ? '#40222a' : '#1f3b2b';
    setTimeout(() => { toast.style.display = 'none'; }, 4000);
  }

  function setLoading(on) {
    submitBtn.disabled = on;
    spinner.style.display = on ? 'inline-block' : 'none';
  }

  function imgEl(src, alt) {
    const img = document.createElement('img');
    img.src = src;
    img.alt = alt;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.style.maxWidth = '100%';
    img.style.border = '1px solid #2a2f3a';
    img.style.borderRadius = '6px';
    return img;
  }

  function renderTeam(payload) {
    const card = document.createElement('div');
    card.className = 'team-card';
    // Removed explicit 'Team A' / 'Team B' headings per design

    const grid = document.createElement('div');
    grid.className = 'grid4';
    if (payload.sca_polar_url) grid.appendChild(imgEl(payload.sca_polar_url, 'SCA Polar'));
    if (payload.gca_polar_url) grid.appendChild(imgEl(payload.gca_polar_url, 'GCA Polar'));
    if (payload.shot_map_url) grid.appendChild(imgEl(payload.shot_map_url, 'Shot Map'));
    if (payload.squad_depth_url) grid.appendChild(imgEl(payload.squad_depth_url, 'Squad Depth'));
    card.appendChild(grid);

    // Pies removed from UI; backend may still return them if enabled via API, render if present
    if (payload.sca_pie_url || payload.gca_pie_url) {
      const pies = document.createElement('div');
      pies.className = 'grid2';
      if (payload.sca_pie_url) pies.appendChild(imgEl(payload.sca_pie_url, 'SCA Pie'));
      if (payload.gca_pie_url) pies.appendChild(imgEl(payload.gca_pie_url, 'GCA Pie'));
      card.appendChild(pies);
    }
    return card;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    results.innerHTML = '';
    // reset layout classes on each submit
    results.classList.remove('two');
    if (cardEl) cardEl.classList.remove('wide');

    // Fixed defaults
    const season = 2025;
    const metric = 'per90';
    // Generation toggles
    const genPolar = document.getElementById('genPolar').checked;
    const genShot = document.getElementById('genShot').checked;
    const genSquad = document.getElementById('genSquad').checked;

    const body = {
      team_a: teamA.value.trim(),
      team_b: teamB.value.trim() || null,
      season: season,
      percentile_metric: metric,
      gen_polar: genPolar,
      gen_shot: genShot,
      gen_squad: genSquad,
    };

    if (!body.team_a && !body.team_b) {
      showToast('Enter at least one team');
      return;
    }

    try {
      setLoading(true);
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || 'Generation failed');
      }
      const data = await res.json();

      // If two teams present, show two columns and widen container
      const twoTeams = !!(data.team_a && data.team_b);
      results.classList.toggle('two', twoTeams);
      if (cardEl) cardEl.classList.toggle('wide', twoTeams);

      if (data.team_a) results.appendChild(renderTeam(data.team_a));
      if (data.team_b) results.appendChild(renderTeam(data.team_b));
    } catch (e) {
      showToast(String(e));
    } finally {
      setLoading(false);
    }
  });
});
