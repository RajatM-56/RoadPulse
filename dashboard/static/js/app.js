document.addEventListener("DOMContentLoaded", () => {
  showScreen("login");
  fetchSummary();
  fetchCameras();
  fetchAnalytics();

  // Scroll listener for dynamic landing nav pill tracking
  window.addEventListener("scroll", () => {
    if (!isLoggedIn) {
      const featuresEl = document.getElementById("features-section");
      if (featuresEl) {
        const rect = featuresEl.getBoundingClientRect();
        if (rect.top <= 250) {
          setActiveNavPill("nav-btn-features");
        } else {
          setActiveNavPill("nav-btn-overview");
        }
      }
    }
  });
});

let currentClip = "fixed_cam_sample";
let currentLayer = "incidents";
let allIncidents = [];
let isLoggedIn = false;

function setActiveNavPill(activeBtnId) {
  const loginNav = document.getElementById("login-nav");
  if (loginNav) {
    const buttons = loginNav.querySelectorAll(".nav-item");
    buttons.forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(activeBtnId);
    if (activeBtn) activeBtn.classList.add("active");
  }
}

function showScreen(screenId, shouldScrollToTop = true) {
  // Hide all screens
  const screens = document.querySelectorAll(".screen-container");
  screens.forEach(s => s.style.display = "none");

  const loginNav = document.getElementById("login-nav");
  const mainNav = document.getElementById("main-nav");
  const headerBtn = document.getElementById("header-btn");

  // Show target screen
  const target = document.getElementById(`screen-${screenId}`);
  if (target) {
    target.style.display = "flex";
  }

  const headerActions = document.querySelector(".header-actions");

  if (screenId === "login") {
    if (loginNav) loginNav.style.display = "flex";
    if (mainNav) mainNav.style.display = "none";
    if (headerActions) headerActions.style.display = "none";
    closeAlertModal();
  } else {
    isLoggedIn = true;
    if (loginNav) loginNav.style.display = "none";
    if (mainNav) mainNav.style.display = "flex";
    if (headerActions) headerActions.style.display = "flex";
    if (headerBtn) headerBtn.innerText = "Sign Out";
  }

  // Update active navbar button state for mainNav
  if (screenId !== "login" && mainNav) {
    const navButtons = mainNav.querySelectorAll(".nav-item");
    navButtons.forEach(btn => btn.classList.remove("active"));
    const activeBtn = Array.from(navButtons).find(b => b.getAttribute("onclick") && b.getAttribute("onclick").includes(screenId));
    if (activeBtn) activeBtn.classList.add("active");
  }

  // Scroll to top if requested
  if (shouldScrollToTop) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function togglePasswordVisibility() {
  const pwInput = document.getElementById("login-password");
  const eyeIcon = document.getElementById("pw-eye-icon");
  if (pwInput) {
    if (pwInput.type === "password") {
      pwInput.type = "text";
      if (eyeIcon) eyeIcon.setAttribute("data-lucide", "eye-off");
    } else {
      pwInput.type = "password";
      if (eyeIcon) eyeIcon.setAttribute("data-lucide", "eye");
    }
    if (window.lucide) lucide.createIcons();
  }
}

function handleHeaderBtnClick() {
  const headerBtn = document.getElementById("header-btn");
  if (headerBtn && headerBtn.innerText === "Sign Out") {
    isLoggedIn = false;
    showScreen("login");
  } else {
    showScreen("login");
    const emailInput = document.querySelector(".glass-input");
    if (emailInput) emailInput.focus();
  }
}

function scrollToLoginSection(section) {
  const loginScreen = document.getElementById("screen-login");
  if (loginScreen && loginScreen.style.display === "none") {
    showScreen("login", false);
  }

  if (section === "overview") {
    setActiveNavPill("nav-btn-overview");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else if (section === "features") {
    setActiveNavPill("nav-btn-features");
    const target = document.getElementById("features-section");
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }
  }
}

function handleLogin(e) {
  e.preventDefault();
  showScreen("dashboard");
}

async function fetchSummary() {
  try {
    const res = await fetch("/api/summary");
    const data = await res.json();

    if (data.phase1_summary && Array.isArray(data.phase1_summary) && data.phase1_summary.length > 0) {
      const avgFps = (data.phase1_summary.reduce((acc, item) => acc + (item.avg_fps || 0), 0) / data.phase1_summary.length).toFixed(1);
      const fpsEl = document.getElementById("stat-fps");
      if (fpsEl) fpsEl.innerText = `${avgFps} FPS`;
    }

    if (data.incidents) {
      allIncidents = data.incidents;
      const incEl = document.getElementById("stat-incidents");
      if (incEl) incEl.innerText = data.incidents.length;

      renderIncidentFeed(data.incidents);
      renderIncidentsTable(data.incidents);
      renderAlertsPage(data.incidents);

      const highSev = data.incidents.find(i => i.severity > 0.75 && i.lifecycle_status === "New");
      if (highSev) {
        showAlertModal(highSev);
      }
    }

    if (data.zones) {
      renderZoneGrid(data.zones);
    }

    loadKinematics(currentClip);
  } catch (err) {
    console.error("Error fetching summary:", err);
  }
}

async function fetchCameras() {
  try {
    const res = await fetch("/api/cameras");
    const data = await res.json();
    if (data.cameras) {
      renderCamerasGrid(data.cameras);
    }
  } catch (e) {
    console.error("Error fetching cameras:", e);
  }
}

async function fetchAnalytics() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();
    if (data) {
      renderAnalyticsView(data);
    }
  } catch (e) {
    console.error("Error fetching analytics:", e);
  }
}

function renderZoneGrid(zones) {
  const container = document.getElementById("zone-grid");
  if (!container) return;

  container.innerHTML = zones.map(z => {
    const statusColor = z.status === "critical" ? "var(--accent-rose)" : (z.status === "warning" ? "var(--accent-amber)" : "var(--accent-emerald)");
    const statusText = z.status === "critical" ? "2 Active Alerts" : (z.status === "warning" ? "1 Warning" : "Normal Flow");

    return `
      <div class="zone-card ${z.status}">
        <div class="zone-header">
          <span class="zone-title">${z.name}</span>
          ${z.coverage_gap ? '<span class="coverage-gap-badge">COVERAGE GAP OVERLAY</span>' : ''}
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between; margin-top: 0.3rem;">
          <span>Zone ID: <strong>${z.id}</strong></span>
          <span style="color: ${statusColor}; font-weight: 700;">● ${statusText}</span>
        </div>
      </div>
    `;
  }).join("");
}

function renderIncidentFeed(incidents) {
  const container = document.getElementById("incident-feed-container");
  if (!container) return;

  if (incidents.length === 0) {
    container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No active traffic incidents flagged.</div>`;
    return;
  }

  container.innerHTML = incidents.map(inc => {
    const sevClass = inc.severity > 0.75 ? "sev-high" : "sev-med";
    const status = inc.lifecycle_status || "New";

    return `
      <div class="event-card" style="border-left-color: ${inc.severity > 0.75 ? 'var(--accent-rose)' : 'var(--accent-amber)'}">
        <div class="event-header">
          <span class="event-type">${inc.type}</span>
          <span class="sev-badge ${sevClass}">Sev: ${inc.severity.toFixed(2)}</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
          <span>Zone: <strong>${inc.zone || 'Grid_Sector'}</strong></span>
          <span>Confidence: <strong>${(inc.confidence * 100).toFixed(0)}%</strong></span>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-dim); display: flex; justify-content: space-between; align-items: center; margin-top: 0.2rem;">
          <span>Timestamp: ${inc.timestamp_s || 0}s &bull; ID: ${inc.incident_id}</span>
          <button class="btn" style="padding: 0.25rem 0.55rem; font-size: 0.72rem;" onclick="viewIncidentDetails('${inc.incident_id}')">Investigate</button>
        </div>

        <div class="lifecycle-controls">
          <span style="font-size: 0.75rem; color: var(--text-muted); margin-right: 0.25rem;">Lifecycle:</span>
          <button class="lifecycle-btn ${status === 'New' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'New')">New</button>
          <button class="lifecycle-btn ${status === 'Acknowledged' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'Acknowledged')">Acknowledged</button>
          <button class="lifecycle-btn ${status === 'Resolved' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'Resolved')">Resolved</button>
        </div>
      </div>
    `;
  }).join("");
}

function renderIncidentsTable(incidents) {
  const tbody = document.getElementById("incidents-table-body");
  if (!tbody) return;

  if (incidents.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">No incidents found in log.</td></tr>`;
    return;
  }

  tbody.innerHTML = incidents.map(inc => {
    const sevClass = inc.severity > 0.75 ? "sev-high" : "sev-med";
    const status = inc.lifecycle_status || "New";

    return `
      <tr>
        <td><strong style="color: var(--primary-blue);">${inc.incident_id}</strong></td>
        <td><strong>${inc.type}</strong></td>
        <td>${inc.zone || 'Grid_Zone'}</td>
        <td><span class="sev-badge ${sevClass}">${inc.severity.toFixed(2)}</span></td>
        <td>${(inc.confidence * 100).toFixed(0)}%</td>
        <td>${inc.timestamp_s || 0}s</td>
        <td>
          <span class="badge" style="background: ${status === 'New' ? '#ffe4e6' : (status === 'Acknowledged' ? '#fef3c7' : '#dcfce7')}; color: ${status === 'New' ? 'var(--accent-rose)' : (status === 'Acknowledged' ? 'var(--accent-amber)' : 'var(--accent-emerald)')}; border: none;">
            ${status}
          </span>
        </td>
        <td>
          <button class="btn btn-primary" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="viewIncidentDetails('${inc.incident_id}')">Investigate</button>
        </td>
      </tr>
    `;
  }).join("");
}

function filterIncidentsTable() {
  const query = (document.getElementById("inc-search").value || "").toLowerCase();
  const typeFilter = document.getElementById("inc-type-filter").value;
  const statusFilter = document.getElementById("inc-status-filter").value;

  const filtered = allIncidents.filter(inc => {
    const matchesQuery = !query || inc.incident_id.toLowerCase().includes(query) || inc.type.toLowerCase().includes(query) || (inc.zone && inc.zone.toLowerCase().includes(query));
    const matchesType = typeFilter === "ALL" || inc.type === typeFilter;
    const matchesStatus = statusFilter === "ALL" || (inc.lifecycle_status || "New") === statusFilter;
    return matchesQuery && matchesType && matchesStatus;
  });

  renderIncidentsTable(filtered);
}

function viewIncidentDetails(incidentId) {
  const inc = allIncidents.find(i => i.incident_id === incidentId);
  if (!inc) return;

  selectedIncidentForDetails = inc;

  document.getElementById("det-title").innerText = `Incident Evidence Investigation — ${inc.incident_id}`;
  document.getElementById("det-type").innerText = inc.type;
  document.getElementById("det-sev").innerText = inc.severity.toFixed(2);
  document.getElementById("det-conf").innerText = `${(inc.confidence * 100).toFixed(0)}%`;
  document.getElementById("det-zone").innerText = inc.zone || "Zone_Sector";
  document.getElementById("det-time").innerText = `${inc.timestamp_s || 0} seconds`;

  const warningBanner = document.getElementById("collision-warning-banner");
  if (inc.type.includes("Collision-Linked")) {
    warningBanner.style.display = "block";
  } else {
    warningBanner.style.display = "none";
  }

  showScreen("details");
}

async function updateCurrentDetailStatus(status) {
  if (selectedIncidentForDetails) {
    await updateLifecycle(selectedIncidentForDetails.incident_id, status);
    alert(`Incident ${selectedIncidentForDetails.incident_id} marked as ${status}.`);
    showScreen("incidents");
  }
}

function renderCamerasGrid(cameras) {
  const container = document.getElementById("cameras-grid-container");
  if (!container) return;

  container.innerHTML = cameras.map(cam => {
    const isOnline = cam.status === "online" || cam.status === "active_flight";
    return `
      <div class="zone-card">
        <div class="zone-header">
          <span class="zone-title">${cam.name}</span>
          <span class="badge" style="background: ${isOnline ? '#dcfce7' : '#f1f5f9'}; color: ${isOnline ? 'var(--accent-emerald)' : 'var(--text-dim)'}; border: none;">
            ${cam.status}
          </span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; flex-direction: column; gap: 0.3rem;">
          <div>Type: <strong>${cam.type}</strong></div>
          <div>Location: ${cam.location}</div>
          <div>Coverage Zone: <strong>${cam.coverage_zone}</strong></div>
          <div>Tracker Engine: <strong>${cam.tracker}</strong></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderAnalyticsView(data) {
  const zContainer = document.getElementById("analytics-zones");
  if (zContainer && data.incidents_by_zone) {
    zContainer.innerHTML = data.incidents_by_zone.map(z => `
      <div style="display: flex; justify-content: space-between; font-size: 0.88rem; padding: 0.5rem; background: #f8fafc; border-radius: 8px; border: 1px solid var(--border-color);">
        <span>${z.zone}</span>
        <strong>${z.count} Incidents</strong>
      </div>
    `).join("");
  }

  const tContainer = document.getElementById("analytics-time");
  if (tContainer && data.incidents_by_time_of_day) {
    tContainer.innerHTML = data.incidents_by_time_of_day.map(t => `
      <div style="display: flex; justify-content: space-between; font-size: 0.88rem; padding: 0.5rem; background: #f8fafc; border-radius: 8px; border: 1px solid var(--border-color);">
        <span>${t.time_window}</span>
        <strong>${t.count} Incidents</strong>
      </div>
    `).join("");
  }
}

function renderAlertsPage(incidents) {
  const container = document.getElementById("alerts-page-container");
  if (!container) return;

  container.innerHTML = incidents.map(inc => `
    <div class="event-card">
      <div class="event-header">
        <span class="event-type">${inc.type}</span>
        <span class="sev-badge ${inc.severity > 0.75 ? 'sev-high' : 'sev-med'}">Severity ${inc.severity.toFixed(2)}</span>
      </div>
      <div style="font-size: 0.82rem; color: var(--text-muted);">
        Location: <strong>${inc.zone}</strong> &bull; Status: <strong>${inc.lifecycle_status || 'New'}</strong>
      </div>
      <div style="font-size: 0.78rem; color: var(--text-dim);">
        Dispatch Stub: Email / SMS Sent to ELCIA Security Station 2
      </div>
    </div>
  `).join("");
}

async function saveSettings(e) {
  e.preventDefault();
  const speed = document.getElementById("set-cong-speed").value;
  const dwell = document.getElementById("set-block-dwell").value;
  const dispatch = document.getElementById("set-dispatch").value;

  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      congestion_speed_thresh: parseFloat(speed),
      blockage_dwell_thresh: parseInt(dwell),
      dispatch_authority: dispatch
    })
  });

  alert("System settings saved successfully.");
}

async function updateLifecycle(incidentId, status) {
  try {
    await fetch("/api/incidents/update-status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incident_id: incidentId, status: status })
    });
    fetchSummary();
  } catch (e) {
    console.error("Failed to update status:", e);
  }
}

function showAlertModal(inc) {
  if (!isLoggedIn) return;
  const modal = document.getElementById("alert-modal");
  const content = document.getElementById("modal-content");
  if (!modal || !content) return;

  content.innerHTML = `
    <p><strong>Incident Type:</strong> <span style="color: var(--accent-rose);">${inc.type}</span></p>
    <p><strong>Location Zone:</strong> ${inc.zone}</p>
    <p><strong>Severity Score:</strong> ${inc.severity.toFixed(2)} (High Priority)</p>
    <p><strong>Detection Confidence:</strong> ${(inc.confidence * 100).toFixed(0)}%</p>
    <p style="margin-top: 0.8rem; font-size: 0.82rem; background: #eff6ff; padding: 0.6rem; border-radius: 8px; border-left: 3px solid var(--primary-blue);">
      <strong>Nearest-Authority Dispatch Routing:</strong><br>
      Automated routing triggered to <em>ELCIA Security Station 2 & Traffic Control</em>.
    </p>
  `;

  modal.style.display = "flex";
}

function closeAlertModal() {
  const modal = document.getElementById("alert-modal");
  if (modal) modal.style.display = "none";
}

async function loadKinematics(clipId) {
  const tbody = document.getElementById("kinematics-body");
  if (!tbody) return;

  try {
    const res = await fetch(`/api/kinematics/${clipId}`);
    if (!res.ok) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No tracking kinematics log found for ${clipId}. Run tracking task below.</td></tr>`;
      return;
    }
    const data = await res.json();
    const keys = Object.keys(data);

    tbody.innerHTML = keys.map(trackId => {
      const item = data[trackId];
      const dirStr = item.final_direction ? `[${item.final_direction[0]}, ${item.final_direction[1]}]` : '[0, 0]';
      return `
        <tr>
          <td><strong style="color: var(--primary-blue);">ID:${trackId}</strong></td>
          <td><span class="badge" style="background: #e0f2fe; color: var(--primary-blue); border: none;">${item.class_name || 'vehicle'}</span></td>
          <td><strong>${item.final_speed_px_s || 0} px/s</strong></td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">${dirStr}</td>
          <td>${item.total_dwell_frames || 0} frames</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--accent-rose);">Error loading kinematics telemetry.</td></tr>`;
  }
}

function changeClipSource() {
  const select = document.getElementById("clip-select");
  currentClip = select.value;
  updateVideoSource();
  loadKinematics(currentClip);
}

function setVideoLayer(layer) {
  currentLayer = layer;
  updateVideoSource();
}

function updateVideoSource() {
  const video = document.getElementById("main-video");
  const titleEl = document.getElementById("player-title");
  
  let src = "";
  let layerLabel = "";
  
  if (currentLayer === "incidents") {
    src = `/media/outputs/incidents/${currentClip}_incidents.mp4`;
    layerLabel = "Incident Alerts Stream";
  } else if (currentLayer === "tracked") {
    src = `/media/outputs/tracks/${currentClip}_tracked.mp4`;
    layerLabel = "Tracked Stream";
  } else if (currentLayer === "annotated") {
    src = `/media/outputs/phase1/${currentClip}_annotated.mp4`;
    layerLabel = "Batch Detections";
  } else {
    src = `/media/data/sample_clips/${currentClip}.mp4`;
    layerLabel = "Raw Feed";
  }
  
  const clipNameLabel = currentClip === "drone_sample" ? "Drone Aerial" : "Fixed Camera Traffic";
  if (titleEl) {
    titleEl.innerText = `${clipNameLabel} — ${layerLabel}`;
  }
  
  if (video) {
    video.src = src;
    video.play().catch(e => console.log("Autoplay prevented:", e));
  }
}

function triggerTask(taskType) {
  const consoleBox = document.getElementById("console-output");
  const line = document.createElement("div");
  line.className = "console-line";
  line.innerText = `[RUNNING] Triggering task: ${taskType} on ${currentClip}...`;
  consoleBox.appendChild(line);
  consoleBox.scrollTop = consoleBox.scrollHeight;

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task: taskType, clip: currentClip })
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    function readChunk() {
      reader.read().then(({ done, value }) => {
        if (done) {
          const finLine = document.createElement("div");
          finLine.className = "console-line success";
          finLine.innerText = `[COMPLETE] Task ${taskType} finished successfully ✓`;
          consoleBox.appendChild(finLine);
          consoleBox.scrollTop = consoleBox.scrollHeight;
          fetchSummary();
          return;
        }
        const text = decoder.decode(value);
        const lines = text.split("\n\n");
        lines.forEach(l => {
          if (l.startsWith("data: ")) {
            const content = l.replace("data: ", "");
            if (content.trim()) {
              const div = document.createElement("div");
              div.className = "console-line";
              div.innerText = content;
              consoleBox.appendChild(div);
            }
          }
        });
        consoleBox.scrollTop = consoleBox.scrollHeight;
        readChunk();
      });
    }
    readChunk();
  }).catch(err => {
    const errLine = document.createElement("div");
    errLine.className = "console-line error";
    errLine.innerText = `[ERROR] Failed to run task: ${err}`;
    consoleBox.appendChild(errLine);
  });
}
