document.addEventListener("DOMContentLoaded", () => {
  const savedLogin = localStorage.getItem("roadpulse_logged_in");
  const savedScreen = localStorage.getItem("roadpulse_current_screen");

  if (savedLogin === "true") {
    isLoggedIn = true;
    showScreen(savedScreen || "dashboard");
  } else {
    isLoggedIn = false;
    showScreen("login");
  }

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

  if (screenId === "incidents") {
    filterIncidentsTable();
  }

  const headerActions = document.querySelector(".header-actions");

  if (screenId === "login") {
    if (loginNav) loginNav.style.display = "flex";
    if (mainNav) mainNav.style.display = "none";
    if (headerActions) headerActions.style.display = "none";
    closeAlertModal();
  } else {
    isLoggedIn = true;
    localStorage.setItem("roadpulse_logged_in", "true");
    localStorage.setItem("roadpulse_current_screen", screenId);
    if (loginNav) loginNav.style.display = "none";
    if (mainNav) mainNav.style.display = "flex";
    if (headerActions) headerActions.style.display = "flex";
    // Hide old text button, show profile avatar
    if (headerBtn) headerBtn.style.display = "none";
    const profileContainer = document.getElementById("user-profile-menu-container");
    if (profileContainer) profileContainer.style.display = "block";
    // Set avatar initial from stored email
    const storedEmail = localStorage.getItem("roadpulse_user_email") || "O";
    const avatarInitial = document.getElementById("profile-avatar-initial");
    if (avatarInitial) avatarInitial.textContent = storedEmail.charAt(0).toUpperCase();
    const emailText = document.getElementById("profile-email-text");
    if (emailText) emailText.textContent = storedEmail;
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
    localStorage.removeItem("roadpulse_logged_in");
    localStorage.removeItem("roadpulse_current_screen");
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
  if (e) e.preventDefault();
  // Capture the email from the login form
  const emailInput = document.querySelector(".glass-input[type='email']");
  const email = emailInput ? emailInput.value.trim() : "operator@elcia.in";
  localStorage.setItem("roadpulse_user_email", email);
  isLoggedIn = true;
  localStorage.setItem("roadpulse_logged_in", "true");
  localStorage.setItem("roadpulse_current_screen", "dashboard");
  showScreen("dashboard");
}

// --- Profile Avatar Dropdown Functions ---
function toggleProfileDropdown(e) {
  if (e) e.stopPropagation();
  const dropdown = document.getElementById("profile-dropdown");
  if (dropdown) {
    const isOpen = dropdown.style.display === "block";
    dropdown.style.display = isOpen ? "none" : "block";
  }
}

function navigateToProfileSettings() {
  const dropdown = document.getElementById("profile-dropdown");
  if (dropdown) dropdown.style.display = "none";
  showScreen("settings");
}

function handleLogout() {
  const dropdown = document.getElementById("profile-dropdown");
  if (dropdown) dropdown.style.display = "none";
  isLoggedIn = false;
  localStorage.removeItem("roadpulse_logged_in");
  localStorage.removeItem("roadpulse_current_screen");
  localStorage.removeItem("roadpulse_user_email");
  // Hide profile container, show sign-in button again
  const profileContainer = document.getElementById("user-profile-menu-container");
  if (profileContainer) profileContainer.style.display = "none";
  const headerBtn = document.getElementById("header-btn");
  if (headerBtn) { headerBtn.style.display = ""; headerBtn.innerText = "Operator Sign In"; }
  showScreen("login");
}

// Close profile dropdown when clicking anywhere outside
document.addEventListener("click", function(e) {
  const container = document.getElementById("user-profile-menu-container");
  const dropdown = document.getElementById("profile-dropdown");
  if (dropdown && container && !container.contains(e.target)) {
    dropdown.style.display = "none";
  }
});

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
      if (highSev && !sessionStorage.getItem("roadpulse_alert_dismissed")) {
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

function formatTimestamp(totalSeconds) {
  if (totalSeconds === undefined || totalSeconds === null || isNaN(totalSeconds)) return "00:00 min";
  const secNum = Math.floor(parseFloat(totalSeconds));
  const minutes = Math.floor(secNum / 60);
  const seconds = secNum % 60;
  const mm = minutes < 10 ? `0${minutes}` : `${minutes}`;
  const ss = seconds < 10 ? `0${seconds}` : `${seconds}`;
  return `${mm}:${ss} min`;
}

function renderIncidentFeed(incidents) {
  const container = document.getElementById("incident-feed-container");
  if (!container) return;

  if (incidents.length === 0) {
    container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No active traffic incidents flagged.</div>`;
    return;
  }

  container.innerHTML = incidents.map(inc => {
    const isHighSev = inc.severity >= 0.75;
    const sevClass = isHighSev ? "sev-high" : "sev-med";
    const sevLabel = isHighSev ? `⚡ Urgency HIGH (${inc.severity.toFixed(2)})` : `⚠️ Urgency MED (${inc.severity.toFixed(2)})`;
    const status = inc.lifecycle_status || "New";

    return `
      <div class="event-card" style="border-left-color: ${isHighSev ? 'var(--accent-rose)' : 'var(--accent-amber)'}">
        <div class="event-header">
          <span class="event-type">${inc.type}</span>
          <span class="sev-badge ${sevClass}">${sevLabel}</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center;">
          <span>Zone <strong>${inc.zone || 'Grid_Sector'}</strong></span>
          <span>Timestamp <strong style="font-family: var(--font-mono);">${formatTimestamp(inc.timestamp_s)}</strong></span>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-dim); display: flex; justify-content: space-between; align-items: center; margin-top: 0.2rem;">
          <span>ID <strong>${inc.incident_id}</strong></span>
          <button class="btn" style="padding: 0.25rem 0.55rem; font-size: 0.72rem;" onclick="viewIncidentDetails('${inc.incident_id}')">Investigate</button>
        </div>

        <div class="lifecycle-controls">
          <span style="font-size: 0.75rem; color: var(--text-muted); margin-right: 0.25rem;">Lifecycle</span>
          <button class="lifecycle-btn ${status === 'New' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'New')">New</button>
          <button class="lifecycle-btn ${status === 'Acknowledged' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'Acknowledged')">Acknowledged</button>
          <button class="lifecycle-btn ${status === 'Resolved' ? 'active' : ''}" onclick="updateLifecycle('${inc.incident_id}', 'Resolved')">Resolved</button>
        </div>
      </div>
    `;
  }).join("");
}

let currentSortColumn = "timestamp_s";
let currentSortDir = "desc";
let selectedIncidentIds = new Set();

function sortTableBy(columnKey) {
  if (currentSortColumn === columnKey) {
    currentSortDir = currentSortDir === "asc" ? "desc" : "asc";
  } else {
    currentSortColumn = columnKey;
    currentSortDir = "asc";
  }

  // Update header sort icons
  const keys = ["incident_id", "type", "zone", "severity", "timestamp_s", "lifecycle_status"];
  keys.forEach(k => {
    const iconEl = document.getElementById(`sort-icon-${k}`);
    if (iconEl) {
      if (k === currentSortColumn) {
        iconEl.innerText = currentSortDir === "asc" ? "▲" : "▼";
        iconEl.style.color = "#ea580c";
      } else {
        iconEl.innerText = "↕";
        iconEl.style.color = "#7c2d12";
      }
    }
  });

  filterIncidentsTable();
}

function renderIncidentsTable(incidents) {
  const tbody = document.getElementById("incidents-table-body");
  if (!tbody) return;

  if (incidents.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2.5rem;">No incidents found matching your criteria.</td></tr>`;
    updateBatchBar();
    return;
  }

  // Sort incidents array
  const sorted = [...incidents].sort((a, b) => {
    let valA = a[currentSortColumn];
    let valB = b[currentSortColumn];

    if (currentSortColumn === "lifecycle_status") {
      valA = a.lifecycle_status || "New";
      valB = b.lifecycle_status || "New";
    }

    if (valA === undefined || valA === null) valA = "";
    if (valB === undefined || valB === null) valB = "";

    if (typeof valA === "string") {
      return currentSortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return currentSortDir === "asc" ? valA - valB : valB - valA;
  });

  tbody.innerHTML = sorted.map(inc => {
    const isHighSev = inc.severity >= 0.75;
    const sevClass = isHighSev ? "sev-high" : "sev-med";
    const sevText = isHighSev ? "⚡ HIGH" : "⚠️ MED";
    const status = inc.lifecycle_status || "New";
    const isChecked = selectedIncidentIds.has(inc.incident_id) ? "checked" : "";

    return `
      <tr style="height: 62px;">
        <td style="text-align: center;" onclick="event.stopPropagation();">
          <input type="checkbox" class="row-checkbox" value="${inc.incident_id}" ${isChecked} onchange="toggleRowSelect('${inc.incident_id}', this.checked)" style="cursor: pointer; width: 15px; height: 15px;">
        </td>
        <td><strong style="color: #ea580c; font-family: var(--font-mono); font-size: 0.88rem;">${inc.incident_id}</strong></td>
        <td><strong>${inc.type}</strong></td>
        <td>${inc.zone || 'Electronics_City_Gate'}</td>
        <td><span class="sev-badge ${sevClass}" style="font-size: 0.74rem;">${sevText} (${inc.severity.toFixed(2)})</span></td>
        <td><span style="font-family: var(--font-mono); font-weight: 600;">${formatTimestamp(inc.timestamp_s)}</span></td>
        <td>
          <select class="status-quick-select status-${status.toLowerCase()}" onchange="updateIncidentStatusInline('${inc.incident_id}', this.value)">
            <option value="New" ${status === 'New' ? 'selected' : ''}>● New</option>
            <option value="Acknowledged" ${status === 'Acknowledged' ? 'selected' : ''}>● Acknowledged</option>
            <option value="Resolved" ${status === 'Resolved' ? 'selected' : ''}>● Resolved</option>
          </select>
        </td>
        <td style="text-align: center;">
          <button class="btn btn-primary" style="padding: 0.4rem 0.85rem; font-size: 0.78rem; border-radius: 8px;" onclick="viewIncidentDetails('${inc.incident_id}')">Investigate</button>
        </td>
      </tr>
    `;
  }).join("");

  updateBatchBar();
}

function filterIncidentsTable() {
  const queryEl = document.getElementById("inc-search");
  const typeEl = document.getElementById("inc-type-filter");
  const statusEl = document.getElementById("inc-status-filter");

  const query = queryEl ? queryEl.value.toLowerCase().trim() : "";
  const typeFilter = typeEl ? typeEl.value : "ALL";
  const statusFilter = statusEl ? statusEl.value : "ALL";

  const filtered = allIncidents.filter(inc => {
    const matchesQuery = !query ||
      inc.incident_id.toLowerCase().includes(query) ||
      inc.type.toLowerCase().includes(query) ||
      (inc.zone && inc.zone.toLowerCase().includes(query));

    const matchesType = typeFilter === "ALL" ||
      inc.type === typeFilter ||
      inc.type.includes(typeFilter) ||
      typeFilter.includes(inc.type.split(" ")[0]);

    const matchesStatus = statusFilter === "ALL" || (inc.lifecycle_status || "New") === statusFilter;
    return matchesQuery && matchesType && matchesStatus;
  });

  renderIncidentsTable(filtered);
}

function updateIncidentStatusInline(incidentId, newStatus) {
  updateLifecycle(incidentId, newStatus);
}

function toggleSelectAllIncidents(headerCheckbox) {
  const isChecked = headerCheckbox.checked;
  const rowCheckboxes = document.querySelectorAll(".row-checkbox");
  rowCheckboxes.forEach(cb => {
    cb.checked = isChecked;
    if (isChecked) {
      selectedIncidentIds.add(cb.value);
    } else {
      selectedIncidentIds.delete(cb.value);
    }
  });
  updateBatchBar();
}

function toggleRowSelect(incidentId, isChecked) {
  if (isChecked) {
    selectedIncidentIds.add(incidentId);
  } else {
    selectedIncidentIds.delete(incidentId);
  }
  const selectAll = document.getElementById("select-all-incidents");
  const rowCheckboxes = document.querySelectorAll(".row-checkbox");
  if (selectAll) {
    selectAll.checked = rowCheckboxes.length > 0 && Array.from(rowCheckboxes).every(cb => cb.checked);
  }
  updateBatchBar();
}

function updateBatchBar() {
  const batchBar = document.getElementById("batch-action-bar");
  const countEl = document.getElementById("batch-count");
  if (batchBar && countEl) {
    const count = selectedIncidentIds.size;
    countEl.innerText = count;
    batchBar.style.display = count > 0 ? "flex" : "none";
  }
}

async function applyBatchStatus(newStatus) {
  if (selectedIncidentIds.size === 0) return;
  const ids = Array.from(selectedIncidentIds);
  for (const id of ids) {
    await updateLifecycle(id, newStatus);
  }
  selectedIncidentIds.clear();
  const selectAll = document.getElementById("select-all-incidents");
  if (selectAll) selectAll.checked = false;
  updateBatchBar();
}

function exportIncidentsCSV() {
  downloadCSV(allIncidents, "RoadPulse_Incident_Register_Full.csv");
}

function exportSelectedCSV() {
  const selectedList = allIncidents.filter(i => selectedIncidentIds.has(i.incident_id));
  if (selectedList.length === 0) return;
  downloadCSV(selectedList, "RoadPulse_Selected_Incidents.csv");
}

function downloadCSV(items, filename) {
  if (!items || items.length === 0) return;

  const headers = ["Incident ID", "Type", "Location Zone", "Urgency", "Timestamp (s)", "Status"];
  const rows = items.map(i => [
    `"${i.incident_id}"`,
    `"${i.type}"`,
    `"${i.zone || 'Electronics_City'}"`,
    i.severity.toFixed(2),
    `${i.timestamp_s || 0}`,
    `"${i.lifecycle_status || 'New'}"`
  ]);

  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function viewIncidentDetails(incidentId) {
  const inc = allIncidents.find(i => i.incident_id === incidentId);
  if (!inc) return;

  selectedIncidentForDetails = inc;

  document.getElementById("det-title").innerText = `Incident Evidence Investigation — ${inc.incident_id}`;
  document.getElementById("det-type").innerText = inc.type;

  const sevEl = document.getElementById("det-sev");
  if (sevEl) {
    const isHigh = inc.severity >= 0.75;
    sevEl.className = `sev-badge ${isHigh ? 'sev-high' : 'sev-med'}`;
    sevEl.innerText = isHigh ? "HIGH" : "MEDIUM";
  }

  document.getElementById("det-zone").innerText = inc.zone || "Zone_Sector";
  document.getElementById("det-time").innerText = formatTimestamp(inc.timestamp_s);

  const camName = inc.incident_id.includes("DRONE") ? "ELCIA Traffic Drone Alpha" : "Phase 1 Gate Junction CCTV";
  const camEl = document.getElementById("det-cam");
  if (camEl) camEl.innerText = camName;

  // Set Video Source and Seek to (-10s to +10s) window
  const videoEl = document.getElementById("det-video");
  const videoLink = document.getElementById("det-dl-video-link");
  const videoPath = inc.incident_id.includes("DRONE") ?
    "/media/outputs/incidents/drone_sample_incidents.mp4" :
    "/media/outputs/incidents/fixed_cam_sample_incidents.mp4";

  if (videoEl) {
    videoEl.src = videoPath;
    const targetSec = inc.timestamp_s || 0;
    const startWindowSec = Math.max(0, targetSec - 10);

    videoEl.addEventListener("loadedmetadata", () => {
      videoEl.currentTime = startWindowSec;
      videoEl.play().catch(e => console.log("Autoplay check:", e));
    }, { once: true });

    // Fallback if metadata already loaded
    videoEl.currentTime = startWindowSec;
    videoEl.play().catch(e => console.log("Autoplay check:", e));

    videoEl.ontimeupdate = () => {
      const badge = document.getElementById("det-current-time-badge");
      if (badge) badge.innerText = formatTimestamp(videoEl.currentTime);
    };
  }

  if (videoLink) {
    videoLink.href = videoPath;
  }

  // AI Logic Description (Human English Translation)
  const logicEl = document.getElementById("det-logic");
  if (logicEl) {
    if (inc.type.includes("Blockage")) {
      logicEl.innerText = `Vehicle remained stationary in active lane for 6 seconds.`;
    } else if (inc.type.includes("Congestion")) {
      logicEl.innerText = `Heavy traffic cluster detected. Zone speed dropped below 15 km/h.`;
    } else if (inc.type.includes("Violation")) {
      logicEl.innerText = `Vehicle moving in wrong direction against designated lane flow.`;
    } else {
      logicEl.innerText = `Abrupt deceleration detected across adjacent vehicles. Verification recommended.`;
    }
  }

  // Dispatch Ticket ID & Status
  const ticketEl = document.getElementById("det-ticket-id");
  const dispatchStatusEl = document.getElementById("det-dispatch-status");
  if (ticketEl) ticketEl.innerText = (1000 + (inc.track_id || 7) * 42).toString();

  const currentStatus = inc.lifecycle_status || "New";

  if (dispatchStatusEl) {
    if (currentStatus === "Resolved") {
      dispatchStatusEl.innerText = "DISPATCH_CLOSED & RESOLVED";
      dispatchStatusEl.style.color = "var(--accent-emerald)";
    } else if (currentStatus === "Acknowledged") {
      dispatchStatusEl.innerText = "ACKNOWLEDGED_IN_PROGRESS";
      dispatchStatusEl.style.color = "var(--accent-amber)";
    } else {
      dispatchStatusEl.innerText = "DISPATCH_ACTIVE";
      dispatchStatusEl.style.color = "#3b82f6";
    }
  }

  // Load Saved Operator Notes
  const savedNotes = localStorage.getItem(`roadpulse_notes_${inc.incident_id}`) || "";
  const notesTextarea = document.getElementById("det-operator-notes");
  if (notesTextarea) notesTextarea.value = savedNotes;

  // Render Action Buttons based on current lifecycle status
  const actionsContainer = document.getElementById("det-actions-container");
  if (actionsContainer) {
    if (currentStatus === "Resolved") {
      actionsContainer.innerHTML = `
        <div style="background: #dcfce7; border: 1px solid #86efac; color: #15803d; padding: 0.75rem 1rem; border-radius: 10px; font-weight: 700; text-align: center; width: 100%; font-size: 0.88rem; display: flex; align-items: center; justify-content: center; gap: 0.4rem;">
          ✅ Incident Marked Resolved & Archived
        </div>
      `;
    } else if (currentStatus === "Acknowledged") {
      actionsContainer.innerHTML = `
        <div style="display: flex; gap: 0.75rem; width: 100%;">
          <div style="background: #fef3c7; border: 1px solid #fde68a; color: #b45309; padding: 0.6rem 0.85rem; border-radius: 8px; font-weight: 700; font-size: 0.8rem; display: flex; align-items: center;">
            ● Acknowledged
          </div>
          <button class="btn" style="flex: 1; justify-content: center; padding: 0.6rem; border-color: var(--accent-emerald); color: var(--accent-emerald);" onclick="updateCurrentDetailStatus('Resolved')">Mark Resolved</button>
        </div>
      `;
    } else {
      actionsContainer.innerHTML = `
        <button class="btn btn-primary" onclick="updateCurrentDetailStatus('Acknowledged')" style="flex: 1; justify-content: center; padding: 0.6rem;">Acknowledge Incident</button>
        <button class="btn" style="flex: 1; justify-content: center; padding: 0.6rem; border-color: var(--accent-emerald); color: var(--accent-emerald);" onclick="updateCurrentDetailStatus('Resolved')">Mark Resolved</button>
      `;
    }
  }

  const warningBanner = document.getElementById("collision-warning-banner");
  if (inc.type.includes("Collision-Linked")) {
    warningBanner.style.display = "block";
  } else {
    warningBanner.style.display = "none";
  }

  showScreen("details");
}

function seekDetailVideo(offsetSeconds) {
  const videoEl = document.getElementById("det-video");
  if (!videoEl || !selectedIncidentForDetails) return;

  const eventTime = selectedIncidentForDetails.timestamp_s || 0;
  const newTime = Math.max(0, eventTime + offsetSeconds);
  videoEl.currentTime = newTime;
  videoEl.play().catch(e => console.log("Play error:", e));
}

function saveCurrentOperatorNotes(notes) {
  if (selectedIncidentForDetails) {
    localStorage.setItem(`roadpulse_notes_${selectedIncidentForDetails.incident_id}`, notes);
  }
}

function exportCurrentIncidentPDF() {
  if (!selectedIncidentForDetails) return;
  const inc = selectedIncidentForDetails;
  const notes = localStorage.getItem(`roadpulse_notes_${inc.incident_id}`) || "None provided";
  const camName = inc.incident_id.includes("DRONE") ? "ELCIA Traffic Drone Alpha" : "Phase 1 Gate Junction CCTV";
  const isHigh = inc.severity >= 0.75;
  const status = inc.lifecycle_status || "New";

  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    alert("Please allow popups to generate PDF report.");
    return;
  }

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>RoadPulse Incident Report — ${inc.incident_id}</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #1e293b; background: #ffffff; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #ea580c; padding-bottom: 15px; margin-bottom: 25px; }
        .title { font-size: 22px; font-weight: 800; color: #7c2d12; }
        .subtitle { font-size: 13px; color: #64748b; margin-top: 4px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; }
        .badge-high { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
        .badge-med { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 20px; }
        th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
        th { background: #fff7ed; color: #9a3412; font-weight: bold; width: 25%; }
        .section-title { font-size: 14px; font-weight: bold; color: #0f172a; margin-top: 22px; margin-bottom: 8px; border-left: 4px solid #ea580c; padding-left: 10px; text-transform: uppercase; }
        .box { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; font-size: 13px; color: #334155; line-height: 1.5; }
        .footer { margin-top: 45px; padding-top: 15px; border-top: 1px solid #cbd5e1; font-size: 11px; color: #94a3b8; text-align: center; }
      </style>
    </head>
    <body>
      <div class="header">
        <div>
          <div class="title">ROADPULSE OPERATIONAL INCIDENT REPORT</div>
          <div class="subtitle">ELCIA Smart City Traffic Management Control Center</div>
        </div>
        <div>
          <div style="font-size: 15px; font-weight: bold; color: #ea580c;">ID: ${inc.incident_id}</div>
          <div style="font-size: 11px; color: #64748b; margin-top: 3px;">Date: ${new Date().toLocaleString()}</div>
        </div>
      </div>

      <table>
        <tr>
          <th>Incident Type</th>
          <td><strong>${inc.type}</strong></td>
          <th>Lifecycle Status</th>
          <td><strong>${status}</strong></td>
        </tr>
        <tr>
          <th>Urgency Level</th>
          <td><span class="badge ${isHigh ? 'badge-high' : 'badge-med'}">${isHigh ? 'HIGH' : 'MEDIUM'} (${inc.severity.toFixed(2)})</span></td>
          <th>Timestamp</th>
          <td><strong>${formatTimestamp(inc.timestamp_s)}</strong></td>
        </tr>
        <tr>
          <th>Location Zone</th>
          <td>${inc.zone || 'Grid Zone'}</td>
          <th>Source Camera</th>
          <td>${camName}</td>
        </tr>
      </table>

      <div class="section-title">AI Kinematics Logic Analysis</div>
      <div class="box">
        ${document.getElementById("det-logic") ? document.getElementById("det-logic").innerText : "Stationary dwell threshold exceeded in active lane."}
      </div>

      <div class="section-title">Automated Dispatch Routing Log</div>
      <div class="box" style="background: #eff6ff; border-color: #bfdbfe; color: #1e40af;">
        Alert dispatched to ELCIA Security Station 2 &bull; Ticket #${1000 + (inc.track_id || 7) * 42} &bull; Status: <strong style="color: var(--accent-emerald);">Active</strong>
      </div>

      <div class="section-title">Control Center Operator Incident Notes</div>
      <div class="box" style="background: #ffffff;">
        ${notes && notes.trim() !== "" ? notes.replace(/\n/g, "<br>") : "No notes logged by operator."}
      </div>

      <div class="footer">
        Confidential &bull; Official ELCIA Smart City Traffic Control Center Forensic Record &bull; RoadPulse 2026
      </div>

      <script>
        window.onload = function() {
          window.print();
        };
      </script>
    </body>
    </html>
  `);
  printWindow.document.close();
}

function showToast(message, type = "success") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.style.cssText = "position: fixed; top: 24px; right: 24px; z-index: 99999; display: flex; flex-direction: column; gap: 10px; pointer-events: none;";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.style.cssText = `
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-left: 5px solid ${type === 'success' ? '#10b981' : '#3b82f6'};
    padding: 0.9rem 1.25rem;
    border-radius: 12px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.12), 0 8px 10px -6px rgba(0, 0, 0, 0.08);
    font-size: 0.88rem;
    font-weight: 600;
    pointer-events: auto;
    transition: all 0.3s ease;
    transform: translateY(-10px);
    opacity: 0;
  `;

  const iconSymbol = type === 'success' ? '✓' : 'ℹ';
  const iconColor = type === 'success' ? '#10b981' : '#3b82f6';
  const iconBg = type === 'success' ? '#dcfce7' : '#eff6ff';

  toast.innerHTML = `
    <div style="background: ${iconBg}; color: ${iconColor}; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem; flex-shrink: 0;">
      ${iconSymbol}
    </div>
    <span style="line-height: 1.4;">${message}</span>
  `;

  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.transform = "translateY(0)";
    toast.style.opacity = "1";
  });

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(-10px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

async function updateCurrentDetailStatus(status) {
  if (selectedIncidentForDetails) {
    const notesTextarea = document.getElementById("det-operator-notes");
    if (notesTextarea && notesTextarea.value.trim()) {
      saveCurrentOperatorNotes(notesTextarea.value.trim());
    }
    await updateLifecycle(selectedIncidentForDetails.incident_id, status);
    showToast(`Incident ${selectedIncidentForDetails.incident_id} marked as ${status}. Notes saved!`, "success");
    showScreen("incidents");
  }
}

function renderCamerasGrid(cameras) {
  const container = document.getElementById("cameras-grid-container");
  if (!container) return;

  // Dynamically calculate camera summary statistics
  const totalCount = cameras.length;
  const dronesCount = cameras.filter(c => c.type.toLowerCase().includes("drone") || c.name.toLowerCase().includes("drone")).length;
  const cctvCount = totalCount - dronesCount;
  const onlineCount = cameras.filter(c => c.status === "online" || c.status === "active_flight").length;
  const uptimePct = totalCount > 0 ? ((onlineCount / totalCount) * 100).toFixed(1) : "100.0";

  const statTotal = document.getElementById("cam-stat-total");
  const statCctv = document.getElementById("cam-stat-cctv");
  const statDrones = document.getElementById("cam-stat-drones");
  const statUptime = document.getElementById("cam-stat-uptime");

  if (statTotal) statTotal.innerText = `${totalCount} Active`;
  if (statCctv) statCctv.innerText = `${cctvCount} Online`;
  if (statDrones) statDrones.innerText = `${dronesCount} Operational`;
  if (statUptime) statUptime.innerText = `${uptimePct}%`;

  container.innerHTML = cameras.map(cam => {
    const isOnline = cam.status === "online" || cam.status === "active_flight";
    const isDrone = cam.type.includes("Drone") || cam.name.includes("Drone");
    const videoPath = isDrone ?
      "/media/outputs/incidents/drone_sample_incidents.mp4" :
      "/media/outputs/incidents/fixed_cam_sample_incidents.mp4";

    const badgeColor = isOnline ? (isDrone ? '#ea580c' : 'var(--accent-emerald)') : 'var(--text-dim)';
    const badgeBg = isOnline ? (isDrone ? '#ffedd5' : '#dcfce7') : '#f1f5f9';
    const statusText = isDrone ? '🛸 FLIGHT ACTIVE' : (isOnline ? '🔴 LIVE' : '⏸️ STANDBY');

    return `
      <div class="zone-card" style="display: flex; flex-direction: column; justify-content: space-between; border-radius: 14px; overflow: hidden; padding: 1.1rem; background: #ffffff; border: 1px solid var(--border-color); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);">
        <div>
          <div class="zone-header" style="margin-bottom: 0.65rem;">
            <span class="zone-title" style="font-size: 0.95rem; font-weight: 700; color: #0f172a;">${cam.name}</span>
            <span class="badge" style="background: ${badgeBg}; color: ${badgeColor}; border: none; font-weight: 700; font-size: 0.72rem; padding: 0.25rem 0.6rem;">
              ${statusText}
            </span>
          </div>

          <!-- Video Stream Preview Box -->
          <div style="position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #0f172a; margin-bottom: 0.85rem; border: 1px solid var(--border-color);">
            <video autoplay loop muted playsinline style="width: 100%; height: 160px; object-fit: cover; opacity: ${isOnline ? '0.92' : '0.4'};">
              <source src="${videoPath}" type="video/mp4">
            </video>
            <div style="position: absolute; top: 8px; left: 8px; background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(4px); color: #ffffff; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.68rem; font-family: var(--font-mono); font-weight: 600;">
              ${cam.tracker || 'AI Object Tracking'}
            </div>
            ${isDrone ? `
              <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(234, 88, 12, 0.9); color: #ffffff; padding: 0.2rem 0.55rem; border-radius: 6px; font-size: 0.68rem; font-weight: 700;">
                ⚡ 94% Battery
              </div>
            ` : ''}
          </div>

          <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 0.9rem;">
            <div>Type <strong>${cam.type}</strong></div>
            <div>Location <strong>${cam.location}</strong></div>
            <div>Coverage Zone <strong>${cam.coverage_zone}</strong></div>
          </div>
        </div>

        <button class="btn btn-primary" onclick="switchActiveStream('${cam.id}', '${cam.name.replace(/'/g, "\\'")}', '${videoPath}')" style="width: 100%; justify-content: center; padding: 0.6rem; font-size: 0.82rem; font-weight: 700; border-radius: 8px;">
          📹 Switch Live Stream to Video Player
        </button>
      </div>
    `;
  }).join("");
}

function switchActiveStream(camId, camName, videoPath) {
  const isDrone = camId.toLowerCase().includes("drone") || camName.toLowerCase().includes("drone");
  currentClip = isDrone ? "drone_sample" : "fixed_cam_sample";

  const player = document.getElementById("video-player");
  const headerText = document.getElementById("video-stream-title");

  if (player) {
    player.src = videoPath;
    player.load();
    player.play().catch(e => console.log("Stream play check:", e));
  }

  if (headerText) {
    headerText.innerText = `${camName.toUpperCase()} | LIVE CONTROL STREAM`;
  }

  // Reload telemetry data for selected clip
  fetchSummary();

  showToast(`📹 Switched live stream feed to ${camName}!`, "info");
  showScreen("dashboard");
}

let chartIncidentDonutInstance = null;
let chartZoneBarsInstance = null;
let chartTimeBarsInstance = null;

function renderAnalyticsView(data) {
  if (typeof Chart === "undefined") return;

  // 1. Spatial Zones Horizontal Bar Chart
  const zoneCtx = document.getElementById("chart-zone-bars");
  if (zoneCtx && data.incidents_by_zone) {
    if (chartZoneBarsInstance) chartZoneBarsInstance.destroy();

    const labels = data.incidents_by_zone.map(z => z.zone.replace(/_/g, " ").replace("Zone A", "Zone A —").replace("Zone B", "Zone B —").replace("Zone C", "Zone C —"));
    const counts = data.incidents_by_zone.map(z => z.count);

    chartZoneBarsInstance = new Chart(zoneCtx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Incident Count",
          data: counts,
          backgroundColor: "#ea580c",
          borderRadius: 6,
          barThickness: 16
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: "#f1f5f9" }, ticks: { precision: 0 } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // 3. Time of Day Peak Traffic Vertical Bar Chart
  const timeCtx = document.getElementById("chart-time-bars");
  if (timeCtx && data.incidents_by_time_of_day) {
    if (chartTimeBarsInstance) chartTimeBarsInstance.destroy();

    const timeLabels = data.incidents_by_time_of_day.map(t => t.time_window);
    const timeCounts = data.incidents_by_time_of_day.map(t => t.count);

    chartTimeBarsInstance = new Chart(timeCtx, {
      type: "bar",
      data: {
        labels: timeLabels,
        datasets: [{
          label: "Traffic Incidents",
          data: timeCounts,
          backgroundColor: "#3b82f6",
          borderRadius: 6,
          barThickness: 22
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { grid: { color: "#f1f5f9" }, ticks: { precision: 0 } }
        }
      }
    });
  }
}

function handleAnalyticsFilterChange(period) {
  const subtext = document.getElementById("analytics-timeframe-subtext");
  if (subtext) {
    subtext.innerHTML = `<span>📅 Timeframe Scope:</span> <strong style="color: #ea580c;">${period}</strong>`;
  }

  showToast(`📊 Analytics dataset updated for ${period}`, "info");

  // Mock Datasets for Today, 7 Days, and 30 Days
  let dataConfig = {
    total: 25,
    c1: "36% • 9 Incidents",
    c2: "20% • 5 Incidents",
    c3: "16% • 4 Incidents",
    c4: "28% • 7 Incidents",
    zones: [14, 9, 2],
    time: [12, 5, 4, 4],
    kpi: {
      recallVal: "78.5%",
      recallSub: "Target: ≥ 70.0% Proposal Target",
      falseVal: "2.1%",
      falseSub: "Minimised Normal Flow Errors",
      latVal: "1.42s",
      latSub: "Kinematic Frame Precision",
      resVal: "94.5%",
      resSub: "ELCIA Control Center Team"
    }
  };

  if (period.includes("7 Days")) {
    dataConfig = {
      total: 142,
      c1: "34% • 48 Incidents",
      c2: "22% • 31 Incidents",
      c3: "14% • 21 Incidents",
      c4: "30% • 42 Incidents",
      zones: [68, 48, 26],
      time: [54, 28, 22, 38],
      kpi: {
        recallVal: "82.1%",
        recallSub: "7-Day Avg (Target: ≥ 70.0%)",
        falseVal: "1.8%",
        falseSub: "7-Day Noise Reduction Baseline",
        latVal: "1.35s",
        latSub: "7-Day Kinematic Precision",
        resVal: "96.2%",
        resSub: "7-Day Operator SLA Clear Rate"
      }
    };
  } else if (period.includes("30 Days")) {
    dataConfig = {
      total: 586,
      c1: "34% • 198 Incidents",
      c2: "21% • 124 Incidents",
      c3: "15% • 89 Incidents",
      c4: "30% • 175 Incidents",
      zones: [274, 196, 116],
      time: [218, 112, 96, 160],
      kpi: {
        recallVal: "84.8%",
        recallSub: "30-Day Monthly Benchmark",
        falseVal: "1.5%",
        falseSub: "Monthly Sustained Noise Floor",
        latVal: "1.28s",
        latSub: "30-Day System Speed Benchmark",
        resVal: "97.8%",
        resSub: "Monthly Control Center Resolution"
      }
    };
  }

  // 1. Update Top KPI Cards
  const kRecallVal = document.getElementById("kpi-recall-val");
  const kRecallSub = document.getElementById("kpi-recall-sub");
  const kFalseVal = document.getElementById("kpi-false-val");
  const kFalseSub = document.getElementById("kpi-false-sub");
  const kLatVal = document.getElementById("kpi-latency-val");
  const kLatSub = document.getElementById("kpi-latency-sub");
  const kResVal = document.getElementById("kpi-resolution-val");
  const kResSub = document.getElementById("kpi-resolution-sub");

  if (kRecallVal) kRecallVal.textContent = dataConfig.kpi.recallVal;
  if (kRecallSub) kRecallSub.textContent = dataConfig.kpi.recallSub;
  if (kFalseVal) kFalseVal.textContent = dataConfig.kpi.falseVal;
  if (kFalseSub) kFalseSub.textContent = dataConfig.kpi.falseSub;
  if (kLatVal) kLatVal.textContent = dataConfig.kpi.latVal;
  if (kLatSub) kLatSub.textContent = dataConfig.kpi.latSub;
  if (kResVal) kResVal.textContent = dataConfig.kpi.resVal;
  if (kResSub) kResSub.textContent = dataConfig.kpi.resSub;

  // 2. Update SVG Donut Badge & Callouts
  const bTotal = document.getElementById("donut-badge-total");
  const cTotal = document.getElementById("donut-center-total");
  const c1 = document.getElementById("donut-callout-1");
  const c2 = document.getElementById("donut-callout-2");
  const c3 = document.getElementById("donut-callout-3");
  const c4 = document.getElementById("donut-callout-4");

  if (bTotal) bTotal.textContent = `${dataConfig.total} Total`;
  if (cTotal) cTotal.textContent = dataConfig.total;
  if (c1) c1.textContent = dataConfig.c1;
  if (c2) c2.textContent = dataConfig.c2;
  if (c3) c3.textContent = dataConfig.c3;
  if (c4) c4.textContent = dataConfig.c4;

  // 3. Update Zone Bar Chart
  if (chartZoneBarsInstance) {
    chartZoneBarsInstance.data.datasets[0].data = dataConfig.zones;
    chartZoneBarsInstance.update();
  }

  // 4. Update Time Bar Chart
  if (chartTimeBarsInstance) {
    chartTimeBarsInstance.data.datasets[0].data = dataConfig.time;
    chartTimeBarsInstance.update();
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

function selectPreset(btn, presetKey) {
  document.querySelectorAll(".preset-btn").forEach(b => {
    b.style.border = "1px solid var(--border-color)";
    b.style.background = "#ffffff";
    b.style.color = "#334155";
  });

  btn.style.border = "2px solid var(--primary-blue)";
  btn.style.background = "#eff6ff";
  btn.style.color = "#1e40af";

  showToast(`⚡ Model inference preset switched to ${presetKey.replace('_', ' ').toUpperCase()}`, "info");
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

  showToast("✅ System calibration settings saved successfully.", "success");
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
    <p><strong>Incident Type</strong> <span style="color: var(--accent-rose);">${inc.type}</span></p>
    <p><strong>Location Zone</strong> ${inc.zone}</p>
    <p><strong>Urgency Level</strong> ${inc.severity.toFixed(2)} (${inc.severity >= 0.75 ? 'HIGH' : 'MED'})</p>
    <p style="margin-top: 0.8rem; font-size: 0.82rem; background: #eff6ff; padding: 0.6rem; border-radius: 8px; border-left: 3px solid var(--primary-blue);">
      <strong>Nearest-Authority Dispatch Routing:</strong><br>
      Automated routing triggered to <em>ELCIA Security Station 2 & Traffic Control</em>.
    </p>
  `;

  modal.style.display = "flex";
}

function closeAlertModal() {
  sessionStorage.setItem("roadpulse_alert_dismissed", "true");
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
