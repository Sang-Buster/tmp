/**
 * Main application - coordinates 3D scene, chat, and data fetching
 * Enhanced with jamming zones and algorithm controls
 */

/**
 * Format ID for display (agent1 -> Agent 1, obstacle_1 -> Obstacle 1)
 */
function formatDisplayName(id) {
  if (!id) return id;
  // Handle patterns like "agent1", "agent_1", "obstacle_1"
  return id
    .replace(/_/g, " ") // Replace underscores with spaces
    .replace(/([a-z])(\d)/gi, "$1 $2") // Add space before numbers
    .replace(/\b\w/g, (c) => c.toUpperCase()); // Capitalize first letter of each word
}

const App = {
  updateInterval: null,
  agentListEl: null,
  jammingListEl: null,
  spoofingListEl: null,
  simApiBase: "", // Defaults to same origin
  simulationRunning: false,
  selectedAgent: null,
  selectedJammingZone: null,
  llmAssistanceEnabled: true, // LLM assistance on by default
  cryptoAuthEnabled: false,
  trailLength: "short", // "short" or "all" for trail visualization

  /**
   * Initialize the application
   */
  async init() {
    console.log("[App] Initializing...");

    // Initialize 3D scene
    const sceneContainer = document.getElementById("scene-container");
    if (sceneContainer && window.Scene3D) {
      Scene3D.init(sceneContainer);
      console.log("[App] 3D scene initialized");
    }

    // Initialize chat
    if (window.Chat) {
      Chat.init();
      console.log("[App] Chat initialized");
    }

    // Get DOM elements
    this.agentListEl = document.getElementById("agent-list");
    this.jammingListEl = document.getElementById("jamming-list");
    this.spoofingListEl = document.getElementById("spoofing-list");

    // Setup event listeners
    this.setupEventListeners();

    // Check health
    await this.checkHealth();

    // Sync LLM assistance state from server
    await syncLLMAssistanceState();

    // Sync crypto auth state from server
    await syncCryptoAuthState();

    // Start polling for updates
    this.startUpdates();

    console.log("[App] Ready");
  },

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Vehicle selection from 3D scene
    window.addEventListener("vehicleSelected", (e) => {
      this.selectedAgent = e.detail.agentId;
      this.selectedJammingZone = null; // Deselect zone when selecting vehicle
      this.refreshAgentList();
      this.refreshJammingList();
    });

    // Jamming zone selection from 3D scene
    window.addEventListener("jammingZoneSelected", (e) => {
      this.selectedJammingZone = e.detail.zoneId;
      this.selectedAgent = null; // Deselect agent when selecting zone
      this.refreshAgentList();
      this.refreshJammingList();
    });

    // Intensity slider
    const intensitySlider = document.getElementById("jam-intensity");
    const intensityValue = document.getElementById("jam-intensity-value");
    if (intensitySlider && intensityValue) {
      intensitySlider.addEventListener("input", (e) => {
        intensityValue.textContent = parseFloat(e.target.value).toFixed(1);
      });
    }

    // Algorithm dropdowns — apply immediately if simulation is running
    const applyAlgorithmChange = async (key, value) => {
      if (!App.simulationRunning) return; // will be picked up on next start
      try {
        await fetch("/simulation/algorithm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ [key]: value }),
        });
        console.log(`[App] Algorithm updated: ${key}=${value}`);
      } catch (err) {
        console.error("[App] Failed to update algorithm:", err);
      }
    };

    const formationSelect = document.getElementById("formation-select");
    if (formationSelect) {
      formationSelect.addEventListener("change", (e) =>
        applyAlgorithmChange("formation", e.target.value)
      );
    }

    const pathAlgoSelect = document.getElementById("path-algo-select");
    if (pathAlgoSelect) {
      pathAlgoSelect.addEventListener("change", (e) =>
        applyAlgorithmChange("path_algorithm", e.target.value)
      );
    }

  },

  /**
   * Check system health
   */
  async checkHealth() {
    try {
      const response = await fetch("/health");
      const data = await response.json();

      this.updateStatus("sim-status", data.simulation_api === "online");
      this.updateStatus("llm-status", data.llm === "ready");
      this.updateStatus("db-status", true);
    } catch (error) {
      console.error("[App] Health check failed:", error);
      this.updateStatus("sim-status", false);
      this.updateStatus("llm-status", false);
    }
  },

  /**
   * Update a status indicator
   */
  updateStatus(id, online) {
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove("bg-success", "bg-destructive", "bg-muted");
      el.classList.add(online ? "bg-success" : "bg-destructive");
    }
  },

  /**
   * Start polling for updates
   */
  startUpdates() {
    this.fetchAgents();
    this.fetchJammingZones();
    this.fetchVisualization();
    this.fetchSpoofingZones();
    this.fetchProtocolStats();
    this.fetchLLMContext();

    this.updateInterval = setInterval(() => {
      this.fetchAgents();
      this.fetchJammingZones();
      this.fetchVisualization();
      this.fetchSpoofingZones();
      this.fetchProtocolStats();
      if (this.simulationRunning) {
        this.fetchSimulationState();
      }
      if (this.llmAssistanceEnabled) {
        this.fetchLLMActivity();
      }
      this.fetchLLMContext();
    }, 500);
  },

  /**
   * Stop polling
   */
  stopUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  },

  /**
   * Fetch agents from API
   */
  async fetchAgents() {
    try {
      const response = await fetch("/agents");
      const data = await response.json();

      if (data.agents) {
        if (window.Scene3D) {
          Scene3D.updateAllVehicles(data.agents);
        }
        this.updateAgentList(data.agents);

        // Update count
        const countEl = document.getElementById("agent-count");
        if (countEl)
          countEl.textContent = `(${Object.keys(data.agents).length})`;
      }
    } catch (error) {
      console.error("[App] Failed to fetch agents:", error);
    }
  },

  /**
   * Fetch jamming zones from API
   */
  async fetchJammingZones() {
    try {
      const response = await fetch("/jamming_zones");

      if (!response.ok) {
        console.warn(
          "[App] Jamming zones API returned error:",
          response.status,
        );
        return; // Don't update on error - keep existing zones
      }

      const data = await response.json();

      // #region agent log
      fetch(
        "http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            location: "app.js:fetchJammingZones",
            message: "Jamming zones API response",
            data: {
              zonesCount: data.zones?.length,
              zones: data.zones?.map((z) => ({ id: z.id, active: z.active })),
              rawResponse: !!data,
            },
            timestamp: Date.now(),
            sessionId: "debug-session",
            hypothesisId: "C",
          }),
        },
      ).catch(() => {});
      // #endregion

      if (data.zones && Array.isArray(data.zones)) {
        if (window.Scene3D) {
          Scene3D.updateAllJammingZones(data.zones);
        }
        this.updateJammingList(data.zones);

        // Update count
        const countEl = document.getElementById("jamming-count");
        if (countEl) countEl.textContent = `(${data.zones.length})`;
      } else {
        console.warn("[App] Invalid jamming zones response:", data);
      }
    } catch (error) {
      console.error("[App] Failed to fetch jamming zones:", error);
      // Don't update zones on error - keep existing ones visible
    }
  },

  /**
   * Fetch visualization data (communication links, waypoints, trails)
   */
  async fetchVisualization() {
    try {
      const response = await fetch(
        `/visualization?trail_length=${this.trailLength}`,
      );
      const data = await response.json();

      if (window.Scene3D && data) {
        Scene3D.updateVisualization(data);
      }

      // Store waypoints for mini-map
      if (window.MiniMap && data.waypoints) {
        MiniMap.setWaypoints(data.waypoints);
      }

      // Store discovered obstacles for mini-map
      if (window.MiniMap && data.discovered_obstacles) {
        MiniMap.setDiscoveredObstacles(data.discovered_obstacles);
      }
    } catch (error) {
      // Visualization endpoint might not be available yet
    }
  },

  /**
   * Fetch LLM activity for chat panel
   */
  async fetchLLMActivity() {
    try {
      const response = await fetch("/llm_activity?limit=5");
      const data = await response.json();

      if (data.activity && data.activity.length > 0 && window.Chat) {
        Chat.processLLMActivity(data.activity);
      }
    } catch (error) {
      // LLM activity endpoint might not be available
    }
  },

  /**
   * Fetch LLM context for context panel
   */
  async fetchLLMContext() {
    try {
      const response = await fetch("/llm_context");
      const data = await response.json();
      this.updateLLMContextPanel(data);
    } catch (error) {
      // LLM context endpoint might not be available
    }
  },

  /**
   * Update LLM Context panel with data
   */
  updateLLMContextPanel(context) {
    // Update count badge
    const countEl = document.getElementById("llm-context-count");
    const assistedCount = context.agents_being_assisted?.length || 0;
    if (countEl) {
      countEl.textContent = `${assistedCount} assisted`;
    }

    // Update assisted agents list
    const assistedEl = document.getElementById("llm-assisted-agents");
    if (assistedEl) {
      if (assistedCount > 0) {
        assistedEl.innerHTML = context.agents_being_assisted
          .map(
            (a) => `
          <div class="flex justify-between items-center py-1 px-2 bg-purple-500/10 rounded">
            <span class="text-purple-300">${formatDisplayName(a.agent_id)}</span>
            <span class="text-muted-foreground">Comm: ${(a.communication_quality * 100).toFixed(0)}%</span>
          </div>
        `,
          )
          .join("");
      } else {
        assistedEl.innerHTML =
          '<div class="text-muted-foreground/60 italic">None</div>';
      }
    }

    // Update active guidance list
    const guidanceEl = document.getElementById("llm-active-guidance");
    if (guidanceEl) {
      const activeGuidance = context.active_guidance || [];
      if (activeGuidance.length > 0) {
        guidanceEl.innerHTML = activeGuidance
          .map(
            (g) => `
          <div class="py-1.5 px-2 bg-secondary/50 rounded">
            <div class="flex justify-between items-center mb-1">
              <span class="text-purple-300 font-medium">${formatDisplayName(g.agent_id)}</span>
              <span class="text-muted-foreground text-xs">${g.expires_in?.toFixed(1) || "?"}s left</span>
            </div>
            <div class="text-muted-foreground/80 text-xs overflow-x-auto whitespace-nowrap pb-1 scrollbar-thin scrollbar-thumb-secondary">${g.reasoning || "No reasoning"}</div>
          </div>
        `,
          )
          .join("");
      } else {
        guidanceEl.innerHTML =
          '<div class="text-muted-foreground/60 italic">None</div>';
      }
    }

    // Update last prompt preview
    const promptEl = document.getElementById("llm-last-prompt");
    if (promptEl) {
      const lastPrompts = context.last_prompts || [];
      if (lastPrompts.length > 0) {
        const latest = lastPrompts[lastPrompts.length - 1];
        promptEl.innerHTML = `
          <div class="text-purple-300 text-xs mb-1">${formatDisplayName(latest.agent_id)} @ ${latest.timestamp?.split("T")[1]?.split(".")[0] || ""}</div>
          <div class="text-purple-200/70 whitespace-pre-wrap">${latest.prompt_preview || "No prompt"}</div>
          ${latest.reasoning ? `<div class="mt-2 text-success/80 border-t border-border/30 pt-1">→ ${latest.reasoning}</div>` : ""}
        `;
      } else {
        promptEl.innerHTML =
          '<span class="text-muted-foreground/60 italic">No prompts yet</span>';
      }
    }

    // Update jamming zones
    const jammingEl = document.getElementById("llm-jamming-zones");
    if (jammingEl) {
      const jammingZones = context.jamming_zones || [];
      if (jammingZones.length > 0) {
        jammingEl.innerHTML = jammingZones
          .map(
            (z) => `
          <div class="flex justify-between items-center py-1 px-2 bg-red-500/10 rounded">
            <span class="text-red-300">${z.id}</span>
            <span class="text-muted-foreground">r=${z.radius}m @ (${z.center.map((c) => c.toFixed(0)).join(", ")})</span>
          </div>
        `,
          )
          .join("");
      } else {
        jammingEl.innerHTML =
          '<div class="text-muted-foreground/60 italic">None</div>';
      }
    }

    // Update spoofing zones
    const spoofingEl = document.getElementById("llm-spoofing-zones");
    if (spoofingEl) {
      const spoofingZones = context.spoofing_zones || [];
      if (spoofingZones.length > 0) {
        spoofingEl.innerHTML = spoofingZones
          .map(
            (z) => `
          <div class="flex justify-between items-center py-1 px-2 bg-orange-500/10 rounded">
            <span class="text-orange-300">${z.id} <span class="text-xs text-muted-foreground">(${z.spoof_type})</span></span>
            <span class="text-muted-foreground">r=${z.radius}m @ (${z.center.map((c) => c.toFixed(0)).join(", ")})</span>
          </div>
        `,
          )
          .join("");
      } else {
        spoofingEl.innerHTML =
          '<div class="text-muted-foreground/60 italic">None</div>';
      }
    }
  },

  /**
   * Fetch simulation state (formation metrics)
   */
  async fetchSimulationState() {
    try {
      const response = await fetch("/simulation/state");
      const data = await response.json();

      this.updateFormationStatus(data.formation);

      // Check if simulation just completed
      checkSimulationCompletion(data.running);

      this.simulationRunning = data.running;

      // Update button states
      document.getElementById("btn-sim-start").disabled = data.running;
      document.getElementById("btn-sim-stop").disabled = !data.running;
    } catch (error) {
      // Simulation endpoint might not exist yet
    }
  },

  /**
   * Update agent list in sidebar
   */
  updateAgentList(agents) {
    if (!this.agentListEl) return;

    // Store data for refresh
    this._lastAgentsData = agents;

    this.agentListEl.innerHTML = "";

    for (const [agentId, data] of Object.entries(agents)) {
      const pos = data.position || [0, 0, 0];
      const jammed = data.jammed || false;
      const speed = data.speed || 0;
      const comm = data.communication_quality || 1;
      const isSelected = this.selectedAgent === agentId;
      const isPhantom = data.is_phantom || false;

      const item = document.createElement("div");
      if (isPhantom) {
        item.className = `p-2 rounded-md border cursor-pointer transition-colors border-purple-500/50 bg-purple-500/10 opacity-70 ${isSelected ? "ring-2 ring-purple-400 bg-purple-500/20" : "hover:bg-purple-500/15"}`;
      } else {
        item.className = `p-2 rounded-md border cursor-pointer transition-colors ${jammed ? "border-red-500/50 bg-red-500/10" : "border-border bg-secondary/30"} ${isSelected ? "ring-2 ring-info bg-info/10" : "hover:bg-secondary/50"}`;
      }
      item.setAttribute("data-agent-id", agentId);
      item.onclick = (e) => {
        if (e.target.tagName !== "BUTTON") {
          this.selectAgent(agentId);
        }
      };

      const statusBadge = isPhantom
        ? '<span class="text-xs px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">PHANTOM</span>'
        : `<span class="text-xs px-1.5 py-0.5 rounded ${jammed ? "bg-red-500/20 text-red-400" : "bg-green-500/20 text-green-400"}">${jammed ? "JAMMED" : "OK"}</span>`;

      const deleteBtn = isPhantom
        ? ''
        : `<button onclick="event.stopPropagation(); deleteAgent('${agentId}')" class="w-5 h-5 rounded hover:bg-red-500/20 flex items-center justify-center text-muted-foreground hover:text-red-400 transition-colors" title="Remove agent">×</button>`;

      const nameClass = isPhantom ? "font-medium text-sm text-purple-300 italic" : "font-medium text-sm";

      item.innerHTML = `
        <div class="flex items-center justify-between mb-1">
          <span class="${nameClass}">${formatDisplayName(agentId)}</span>
          <div class="flex items-center gap-1">
            ${statusBadge}
            ${deleteBtn}
          </div>
        </div>
        <div class="text-xs text-muted-foreground space-y-0.5">
          <div class="flex justify-between"><span>Pos:</span><span>(${pos[0].toFixed(0)}, ${pos[1].toFixed(0)}, ${pos[2].toFixed(0)})</span></div>
          <div class="flex justify-between"><span>Speed:</span><span>${speed.toFixed(1)} m/s</span></div>
          <div class="flex justify-between"><span>Comm:</span><span>${(comm * 100).toFixed(0)}%</span></div>
        </div>
      `;
      this.agentListEl.appendChild(item);
    }

    // Update count
    const countEl = document.getElementById("agent-count");
    if (countEl) countEl.textContent = Object.keys(agents).length;
  },

  /**
   * Update jamming zone list
   */
  updateJammingList(zones) {
    if (!this.jammingListEl) return;

    // Store data for refresh
    this._lastJammingData = zones;

    this.jammingListEl.innerHTML = "";

    if (zones.length === 0) {
      this.jammingListEl.innerHTML =
        '<p class="text-xs text-muted-foreground py-2">No jamming zones</p>';
      return;
    }

    for (const zone of zones) {
      const center = zone.center || [0, 0, 0];
      const isSelected = this.selectedJammingZone === zone.id;

      const item = document.createElement("div");
      item.className = `p-2 rounded-md border cursor-pointer transition-colors ${isSelected ? "border-red-500 bg-red-500/20 ring-2 ring-red-500/50" : "border-red-500/30 bg-red-500/5 hover:bg-red-500/10"} ${zone.active ? "" : "opacity-50"}`;
      item.onclick = (e) => {
        // Don't select if clicking the delete button
        if (e.target.tagName !== "BUTTON") {
          this.selectJammingZone(zone.id);
        }
      };

      item.innerHTML = `
        <div class="flex items-center justify-between mb-1">
          <span class="font-medium text-sm text-red-400">${formatDisplayName(zone.id)}</span>
          <button onclick="event.stopPropagation(); deleteJammingZone('${zone.id}')" class="w-5 h-5 rounded hover:bg-red-500/20 flex items-center justify-center text-muted-foreground hover:text-red-400 transition-colors">×</button>
        </div>
        <div class="text-xs text-muted-foreground">
          <div>Center: (${center[0].toFixed(0)}, ${center[1].toFixed(0)}, ${center[2].toFixed(0)})</div>
          <div>Radius: ${zone.radius}m</div>
        </div>
      `;
      this.jammingListEl.appendChild(item);
    }

    // Update count
    const countEl = document.getElementById("jamming-count");
    if (countEl) countEl.textContent = zones.length;
  },

  /**
   * Update formation status display
   */
  updateFormationStatus(formation) {
    const statusEl = document.getElementById("formation-status");
    if (!statusEl || !formation) return;

    statusEl.classList.remove("hidden");
    statusEl.innerHTML = `
      <div class="flex justify-between">
        <span>Status:</span>
        <span class="${formation.converged ? "text-green-400" : "text-yellow-400"}">
          ${formation.converged ? "✓ Converged" : "⟳ Forming"}
        </span>
      </div>
      <div class="flex justify-between">
        <span>Comm Quality:</span>
        <span>${(formation.average_comm_quality * 100).toFixed(1)}%</span>
      </div>
      <div class="flex justify-between">
        <span>Avg Distance:</span>
        <span>${formation.average_neighbor_distance.toFixed(2)} m</span>
      </div>
    `;
  },

  /**
   * Select an agent (called when clicking in sidebar)
   */
  selectAgent(agentId) {
    this.selectedAgent = agentId;
    this.selectedJammingZone = null;

    // Update 3D scene selection
    if (window.Scene3D) {
      Scene3D.selectVehicle(agentId);
    }

    // Refresh lists to show selection
    this.refreshAgentList();
    this.refreshJammingList();
  },

  /**
   * Select a jamming zone (called when clicking in sidebar)
   */
  selectJammingZone(zoneId) {
    this.selectedJammingZone = zoneId;
    this.selectedAgent = null;

    // Update 3D scene selection
    if (window.Scene3D) {
      Scene3D.selectJammingZone(zoneId);
    }

    // Refresh lists to show selection
    this.refreshAgentList();
    this.refreshJammingList();
  },

  /**
   * Refresh agent list (re-render with current selection state)
   */
  refreshAgentList() {
    if (this._lastAgentsData) {
      this.updateAgentList(this._lastAgentsData);
    }
  },

  /**
   * Refresh jamming list (re-render with current selection state)
   */
  refreshJammingList() {
    if (this._lastJammingData) {
      this.updateJammingList(this._lastJammingData);
    }
  },

  /**
   * Fetch spoofing zones from API
   */
  async fetchSpoofingZones() {
    try {
      const response = await fetch("/spoofing_zones");
      if (!response.ok) return;
      const data = await response.json();

      if (data.zones && Array.isArray(data.zones)) {
        if (window.Scene3D) {
          Scene3D.updateAllSpoofingZones(data.zones);
        }
        // Render phantom agents from spoofing zone data
        if (data.phantom_agents && window.Scene3D) {
          Scene3D.updatePhantomAgents(data.phantom_agents);
        }
        this.updateSpoofingList(data.zones);
        const countEl = document.getElementById("spoofing-count");
        if (countEl) countEl.textContent = data.zones.length;
      }
    } catch (error) {
      // Spoofing endpoint might not be available
    }
  },

  /**
   * Fetch MAVLink protocol stats
   */
  async fetchProtocolStats() {
    try {
      const response = await fetch("/protocol_stats");
      if (!response.ok) return;
      const data = await response.json();
      this.updateProtocolStats(data);
    } catch (error) {
      // Protocol stats endpoint might not be available
    }
  },

  /**
   * Update spoofing zone list in sidebar
   */
  updateSpoofingList(zones) {
    if (!this.spoofingListEl) return;
    this._lastSpoofingData = zones;
    this.spoofingListEl.innerHTML = "";

    if (zones.length === 0) {
      this.spoofingListEl.innerHTML =
        '<p class="text-xs text-muted-foreground py-2">No spoofing zones</p>';
      return;
    }

    const typeColors = {
      phantom: "purple",
      position_falsification: "pink",
      coordinate: "violet",
    };

    const typeLabels = {
      phantom: "Phantom",
      position_falsification: "Pos Falsify",
      coordinate: "Coordinate",
    };

    for (const zone of zones) {
      const center = zone.center || [0, 0, 0];
      const color = typeColors[zone.spoof_type] || "purple";
      const label = typeLabels[zone.spoof_type] || zone.spoof_type;

      const item = document.createElement("div");
      item.className = `p-2 rounded-md border border-${color}-500/30 bg-${color}-500/5 hover:bg-${color}-500/10 cursor-pointer transition-colors ${zone.active ? "" : "opacity-50"}`;
      item.onclick = (e) => {
        if (e.target.tagName !== "BUTTON") {
          if (window.Scene3D) Scene3D.selectSpoofingZone(zone.id);
        }
      };

      item.innerHTML = `
        <div class="flex items-center justify-between mb-1">
          <span class="font-medium text-sm text-purple-300">${formatDisplayName(zone.id)}</span>
          <div class="flex items-center gap-1">
            <span class="text-xs px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">${label}</span>
            <button onclick="event.stopPropagation(); deleteSpoofingZone('${zone.id}')" class="w-5 h-5 rounded hover:bg-purple-500/20 flex items-center justify-center text-muted-foreground hover:text-purple-400 transition-colors">×</button>
          </div>
        </div>
        <div class="text-xs text-muted-foreground">
          <div>Center: (${center[0].toFixed(0)}, ${center[1].toFixed(0)}, ${center[2].toFixed(0)})</div>
          <div>Radius: ${zone.radius}m</div>
        </div>
      `;
      this.spoofingListEl.appendChild(item);
    }
  },

  /**
   * Update MAVLink protocol stats panel
   */
  updateProtocolStats(data) {
    const statsEl = document.getElementById("protocol-stats");
    if (!statsEl) return;

    const hasActivity = data.mavlink && (data.mavlink.messages_sent > 0 || data.spoofing_zones_active > 0);
    if (hasActivity) {
      statsEl.classList.remove("hidden");
    }

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    if (data.mavlink) {
      set("proto-sent", data.mavlink.messages_sent);
      set("proto-received", data.mavlink.messages_received);
      set("proto-dropped", data.mavlink.messages_dropped);
      set("proto-spoofed", data.mavlink.messages_spoofed_injected);
      set("proto-rejected", data.mavlink.messages_crypto_rejected);
    }

    // Update crypto status stats panel
    const cryptoStatus = document.getElementById("crypto-status");
    if (cryptoStatus) {
      if (data.crypto_auth_enabled) {
        cryptoStatus.classList.remove("hidden");
        set("crypto-accepted", data.crypto?.accepted || 0);
        set("crypto-rejected-count", data.crypto?.rejected || 0);
        set("crypto-sign-time", data.crypto?.avg_sign_time_us || 0);
        set("crypto-verify-time", data.crypto?.avg_verify_time_us || 0);
      } else {
        cryptoStatus.classList.add("hidden");
      }
    }

    // Sync toggle button visual state with server
    const toggle = document.getElementById("crypto-toggle");
    const knob = document.getElementById("crypto-toggle-knob");
    if (toggle && knob) {
      const serverEnabled = !!data.crypto_auth_enabled;
      const uiEnabled = toggle.dataset.enabled === "true";
      if (serverEnabled !== uiEnabled) {
        toggle.dataset.enabled = serverEnabled.toString();
        App.cryptoAuthEnabled = serverEnabled;
        if (serverEnabled) {
          toggle.classList.remove("bg-muted");
          toggle.classList.add("bg-green-500");
          knob.classList.add("translate-x-5");
          knob.style.transform = "translateX(1.25rem)";
        } else {
          toggle.classList.remove("bg-green-500");
          toggle.classList.add("bg-muted");
          knob.classList.remove("translate-x-5");
          knob.style.transform = "translateX(0)";
        }
      }
    }

    // Sync algorithm dropdown
    if (data.crypto_algorithm) {
      const algoSelect = document.getElementById("crypto-algo-select");
      if (algoSelect && algoSelect.value !== data.crypto_algorithm) {
        algoSelect.value = data.crypto_algorithm;
      }
    }

    // Update header indicators
    const cryptoIndicator = document.getElementById("crypto-indicator");
    if (cryptoIndicator) {
      if (data.crypto_auth_enabled) {
        cryptoIndicator.classList.remove("hidden");
      } else {
        cryptoIndicator.classList.add("hidden");
      }
    }
  },
};

// ============================================================================
// PANEL TOGGLE
// ============================================================================

function togglePanel(panelId) {
  const content = document.getElementById(`${panelId}-content`);
  const chevron = document.getElementById(`${panelId}-chevron`);

  if (content) {
    content.classList.toggle("hidden");
    if (chevron) {
      chevron.classList.toggle("rotate-180");
    }
  }
}

// ============================================================================
// JAMMING ZONE MANAGEMENT
// ============================================================================

function showAddJammingModal() {
  const modal = document.getElementById("jamming-modal");
  if (modal) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
  }
}

function hideAddJammingModal() {
  const modal = document.getElementById("jamming-modal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  }
}

async function createJammingZone() {
  const x = parseFloat(document.getElementById("jam-x").value) || 0;
  const y = parseFloat(document.getElementById("jam-y").value) || 0;
  const z = parseFloat(document.getElementById("jam-z").value) || 0;
  const radius = parseFloat(document.getElementById("jam-radius").value) || 3;
  const intensity =
    parseFloat(document.getElementById("jam-intensity").value) || 1;
  const obstacleType =
    document.getElementById("jam-obstacle-type")?.value || "low_jam";

  try {
    const response = await fetch("/jamming_zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        center: [x, y, z],
        radius: radius,
        intensity: intensity,
        active: true,
        obstacle_type: obstacleType,
      }),
    });

    if (response.ok) {
      hideAddJammingModal();
      App.fetchJammingZones();
    } else {
      const error = await response.json();
      alert(`Failed to create zone: ${error.detail}`);
    }
  } catch (error) {
    console.error("Failed to create jamming zone:", error);
  }
}

async function deleteJammingZone(zoneId) {
  try {
    const response = await fetch(`/jamming_zones/${zoneId}`, {
      method: "DELETE",
    });

    if (response.ok) {
      App.fetchJammingZones();
    }
  } catch (error) {
    console.error("Failed to delete jamming zone:", error);
  }
}

// ============================================================================
// SPOOFING ZONE MANAGEMENT
// ============================================================================

function showAddSpoofingModal() {
  const modal = document.getElementById("spoofing-modal");
  if (!modal) {
    console.error("[App] spoofing-modal element not found");
    return;
  }
  modal.classList.remove("hidden");
  modal.classList.add("flex");
  // Sync attack type from algorithm control dropdown, but override "none" with "phantom"
  const algoType = document.getElementById("spoof-type-select")?.value;
  const modalType = document.getElementById("spoof-modal-type");
  if (modalType) {
    modalType.value = (algoType && algoType !== "none") ? algoType : "phantom";
  }
  if (window.lucide) lucide.createIcons();
}

function hideAddSpoofingModal() {
  const modal = document.getElementById("spoofing-modal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  }
}

async function createSpoofingZone() {
  const x = parseFloat(document.getElementById("spoof-x").value) || 0;
  const y = parseFloat(document.getElementById("spoof-y").value) || 0;
  const z = parseFloat(document.getElementById("spoof-z").value) || 0;
  const radius = parseFloat(document.getElementById("spoof-radius").value) || 15;
  const spoofType = document.getElementById("spoof-modal-type")?.value || "phantom";
  const phantomCount = parseInt(document.getElementById("spoof-phantom-count")?.value) || 2;
  const magnitude = parseFloat(document.getElementById("spoof-magnitude")?.value) || 8;

  console.log("[App] Creating spoofing zone:", { center: [x, y, z], radius, spoofType });

  try {
    const response = await fetch("/spoofing_zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        center: [x, y, z],
        radius: radius,
        spoof_type: spoofType,
        phantom_count: phantomCount,
        falsification_magnitude: magnitude,
        coordinate_vector: [10, 10, 0],
        active: true,
      }),
    });

    const result = await response.json();
    console.log("[App] Spoofing zone response:", result);

    if (response.ok && result.success) {
      hideAddSpoofingModal();
      App.fetchSpoofingZones();
      // Auto-expand spoofing panel to show the new zone
      const content = document.getElementById("spoofing-content");
      if (content && content.classList.contains("hidden")) {
        togglePanel("spoofing");
      }
    } else {
      alert(`Failed to create spoofing zone: ${result.detail || result.error || "Unknown error"}`);
    }
  } catch (error) {
    console.error("Failed to create spoofing zone:", error);
    alert(`Network error creating spoofing zone: ${error.message}`);
  }
}

async function deleteSpoofingZone(zoneId) {
  try {
    const response = await fetch(`/spoofing_zones/${zoneId}`, {
      method: "DELETE",
    });
    if (response.ok) {
      App.fetchSpoofingZones();
    }
  } catch (error) {
    console.error("Failed to delete spoofing zone:", error);
  }
}

// ============================================================================
// CRYPTO AUTH MANAGEMENT
// ============================================================================

async function toggleCryptoAuth() {
  const toggle = document.getElementById("crypto-toggle");
  const knob = document.getElementById("crypto-toggle-knob");
  if (!toggle) {
    console.error("[App] crypto-toggle element not found");
    return;
  }

  const currentEnabled = toggle.dataset.enabled === "true";
  const newEnabled = !currentEnabled;
  const algo = document.getElementById("crypto-algo-select")?.value || "hmac_sha256";

  console.log(`[App] toggleCryptoAuth: ${currentEnabled} -> ${newEnabled}, algo=${algo}`);

  try {
    const response = await fetch("/simulation/crypto_auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: newEnabled, algorithm: algo }),
    });

    const result = await response.json();
    console.log("[App] Crypto auth response:", result);

    if (response.ok && result.success) {
      toggle.dataset.enabled = newEnabled.toString();
      App.cryptoAuthEnabled = newEnabled;

      if (newEnabled) {
        toggle.classList.remove("bg-muted");
        toggle.classList.add("bg-green-500");
        if (knob) {
          knob.classList.add("translate-x-5");
          knob.style.transform = "translateX(1.25rem)";
        }
      } else {
        toggle.classList.remove("bg-green-500");
        toggle.classList.add("bg-muted");
        if (knob) {
          knob.classList.remove("translate-x-5");
          knob.style.transform = "translateX(0)";
        }
      }

      const statusEl = document.getElementById("crypto-status");
      if (statusEl) {
        if (newEnabled) statusEl.classList.remove("hidden");
        else statusEl.classList.add("hidden");
      }

      console.log(`[App] Crypto auth ${newEnabled ? "enabled" : "disabled"} (${algo})`);
    } else {
      console.error("[App] Crypto toggle failed:", result);
      alert(`Crypto auth toggle failed: ${result.error || result.detail || "Unknown error"}`);
    }
  } catch (error) {
    console.error("Failed to toggle crypto auth:", error);
    alert(`Network error toggling crypto auth: ${error.message}`);
  }
}

async function setCryptoAlgorithm(algo) {
  // Always send the algorithm change - the server stores it for when crypto is enabled
  const isEnabled = App.cryptoAuthEnabled || false;
  console.log(`[App] setCryptoAlgorithm: ${algo}, currently enabled=${isEnabled}`);

  try {
    const response = await fetch("/simulation/crypto_auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: isEnabled, algorithm: algo }),
    });
    const result = await response.json();
    console.log(`[App] Crypto algorithm set to: ${algo}`, result);
  } catch (error) {
    console.error("Failed to set crypto algorithm:", error);
  }
}

// ============================================================================
// AGENT MANAGEMENT
// ============================================================================

function showAddAgentModal() {
  const modal = document.getElementById("agent-modal");
  if (modal) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    // Update lucide icons
    if (window.lucide) lucide.createIcons();
  }
}

function hideAddAgentModal() {
  const modal = document.getElementById("agent-modal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  }
}

function randomizeAgentCoords() {
  // Random coordinates within bounds (roughly near the starting area)
  const x = Math.round((Math.random() * 60 - 30) * 10) / 10; // -30 to 30
  const y = Math.round(Math.random() * 80 * 10) / 10; // 0 to 80
  const z = Math.round(Math.random() * 20 * 10) / 10; // 0 to 20

  document.getElementById("agent-x").value = x;
  document.getElementById("agent-y").value = y;
  document.getElementById("agent-z").value = z;
}

async function createAgent() {
  const x = parseFloat(document.getElementById("agent-x").value) || 0;
  const y = parseFloat(document.getElementById("agent-y").value) || 50;
  const z = parseFloat(document.getElementById("agent-z").value) || 0;

  try {
    const response = await fetch("/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, z }),
    });

    if (response.ok) {
      hideAddAgentModal();
      App.fetchAgents();
      const data = await response.json();
      console.log("[App] Created agent:", data.message);
    } else {
      const error = await response.json();
      alert(`Failed to create agent: ${error.detail || response.statusText}`);
    }
  } catch (error) {
    console.error("Failed to create agent:", error);
    alert(`Failed to create agent: ${error.message}`);
  }
}

async function createAgentRandom() {
  // Create agent at random coordinates
  const x = Math.round((Math.random() * 60 - 30) * 10) / 10;
  const y = Math.round(Math.random() * 80 * 10) / 10;
  const z = Math.round(Math.random() * 20 * 10) / 10;

  try {
    const response = await fetch("/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, z }),
    });

    if (response.ok) {
      hideAddAgentModal();
      App.fetchAgents();
      const data = await response.json();
      console.log("[App] Created agent:", data.message);
    } else {
      const error = await response.json();
      alert(`Failed to create agent: ${error.detail || response.statusText}`);
    }
  } catch (error) {
    console.error("Failed to create agent:", error);
    alert(`Failed to create agent: ${error.message}`);
  }
}

async function deleteAgent(agentId) {
  if (
    !confirm(`Are you sure you want to remove ${formatDisplayName(agentId)}?`)
  ) {
    return;
  }

  try {
    const response = await fetch(`/agents/${agentId}`, {
      method: "DELETE",
    });

    if (response.ok) {
      App.fetchAgents();
      // Also remove from 3D scene
      if (window.Scene3D) {
        Scene3D.removeVehicle(agentId);
      }
      console.log(`[App] Deleted agent ${agentId}`);
    } else {
      const error = await response
        .json()
        .catch(() => ({ detail: response.statusText }));
      alert(
        `Failed to delete agent: ${error.detail || "Server error - try restarting the server"}`,
      );
    }
  } catch (error) {
    console.error("Failed to delete agent:", error);
    alert(`Failed to delete agent: ${error.message}`);
  }
}

// ============================================================================
// SIMULATION CONTROL
// ============================================================================

async function startSimulation() {
  const formation = document.getElementById("formation-select").value;
  const pathAlgo = document.getElementById("path-algo-select").value;
  const cryptoEnabled = document.getElementById("crypto-toggle")?.dataset?.enabled === "true";
  const cryptoAlgo = document.getElementById("crypto-algo-select")?.value || "hmac_sha256";

  try {
    const response = await fetch("/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        formation: formation,
        path_algorithm: pathAlgo,
        crypto_auth: cryptoEnabled,
        crypto_algorithm: cryptoAlgo,
      }),
    });

    if (response.ok) {
      App.simulationRunning = true;
      document.getElementById("btn-sim-start").disabled = true;
      document.getElementById("btn-sim-stop").disabled = false;
      console.log("[App] Simulation started");
    }
  } catch (error) {
    console.error("Failed to start simulation:", error);
  }
}

async function stopSimulation() {
  try {
    const response = await fetch("/simulation/stop", {
      method: "POST",
    });

    if (response.ok) {
      App.simulationRunning = false;
      document.getElementById("btn-sim-start").disabled = false;
      document.getElementById("btn-sim-stop").disabled = true;
      // Hide mini-map
      if (window.MiniMap) MiniMap.setVisible(false);
      console.log("[App] Simulation stopped");

      // Fetch and show results modal (user can close and resume later)
      try {
        const resultsResponse = await fetch("/simulation/results");
        if (resultsResponse.ok) {
          const results = await resultsResponse.json();
          if (results && results.steps > 0) {
            window._lastSimResults = results;
            showResultsModalWithData(results);
          }
        }
      } catch (resultsError) {
        console.error("Failed to fetch results:", resultsError);
      }
    }
  } catch (error) {
    console.error("Failed to stop simulation:", error);
  }
}

async function resetSimulation() {
  try {
    // Check if there are results to show BEFORE resetting
    // (reset will clear the results on the backend)
    const resultsResponse = await fetch("/simulation/results");
    const results = resultsResponse.ok ? await resultsResponse.json() : null;
    const hasResults = results && results.steps > 0;

    const response = await fetch("/simulation/reset", {
      method: "POST",
    });

    if (response.ok) {
      App.simulationRunning = false;
      document.getElementById("btn-sim-start").disabled = false;
      document.getElementById("btn-sim-stop").disabled = true;
      // Hide mini-map
      if (window.MiniMap) MiniMap.setVisible(false);
      console.log("[App] Simulation reset");

      // Show results modal if simulation had run
      if (hasResults) {
        // Store results temporarily and show modal
        window._lastSimResults = results;
        showResultsModalWithData(results);
      }
    }
  } catch (error) {
    console.error("Failed to reset simulation:", error);
  }
}

// ============================================================================
// LLM ASSISTANCE CONTROL
// ============================================================================

/**
 * Toggle LLM assistance on/off
 */
async function toggleLLMAssistance() {
  const newState = !App.llmAssistanceEnabled;

  try {
    const response = await fetch("/simulation/llm_assistance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: newState }),
    });

    if (response.ok) {
      App.llmAssistanceEnabled = newState;
      updateLLMToggleUI(newState);
      console.log(`[App] LLM assistance ${newState ? "enabled" : "disabled"}`);
    }
  } catch (error) {
    console.error("Failed to toggle LLM assistance:", error);
  }
}

/**
 * Update the LLM toggle UI to reflect current state
 */
function updateLLMToggleUI(enabled) {
  const toggle = document.getElementById("llm-toggle");
  const knob = document.getElementById("llm-toggle-knob");

  if (toggle && knob) {
    toggle.dataset.enabled = enabled;

    if (enabled) {
      toggle.classList.remove("bg-muted");
      toggle.classList.add("bg-success");
      knob.classList.add("translate-x-5");
      knob.classList.remove("translate-x-0");
    } else {
      toggle.classList.remove("bg-success");
      toggle.classList.add("bg-muted");
      knob.classList.remove("translate-x-5");
      knob.classList.add("translate-x-0");
    }
  }
}

/**
 * Fetch and sync LLM assistance state from server
 */
async function syncLLMAssistanceState() {
  try {
    const response = await fetch("/simulation/llm_assistance");
    if (response.ok) {
      const data = await response.json();
      App.llmAssistanceEnabled = data.enabled;
      updateLLMToggleUI(data.enabled);
    }
  } catch (error) {
    // Ignore errors - server might not be ready
  }
}

/**
 * Fetch crypto auth state from server and update toggle UI
 */
async function syncCryptoAuthState() {
  try {
    const response = await fetch("/protocol_stats");
    if (response.ok) {
      const data = await response.json();
      App.cryptoAuthEnabled = !!data.crypto_auth_enabled;
      const toggle = document.getElementById("crypto-toggle");
      const knob = document.getElementById("crypto-toggle-knob");
      if (toggle && knob) {
        toggle.dataset.enabled = App.cryptoAuthEnabled.toString();
        if (App.cryptoAuthEnabled) {
          toggle.classList.remove("bg-muted");
          toggle.classList.add("bg-green-500");
          knob.classList.add("translate-x-5");
          knob.style.transform = "translateX(1.25rem)";
        } else {
          toggle.classList.remove("bg-green-500");
          toggle.classList.add("bg-muted");
          knob.classList.remove("translate-x-5");
          knob.style.transform = "translateX(0)";
        }
      }
      if (data.crypto_algorithm) {
        const algoSelect = document.getElementById("crypto-algo-select");
        if (algoSelect) algoSelect.value = data.crypto_algorithm;
      }
    }
  } catch (error) {
    // Ignore errors - server might not be ready
  }
}

// ============================================================================
// SIMULATION RESULTS MODAL
// ============================================================================

let resultsChart = null;
let lastSimulationRunning = false;

/**
 * Show simulation results modal
 */
async function showResultsModal() {
  const modal = document.getElementById("results-modal");
  if (modal) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    lucide.createIcons();
    await fetchAndDisplayResults();
  }
}

/**
 * Show simulation results modal with pre-fetched data
 * Used when reset is clicked to show results before they're cleared
 */
function showResultsModalWithData(results) {
  const modal = document.getElementById("results-modal");
  if (modal && results) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    lucide.createIcons();
    displayResults(results);
  }
}

/**
 * Hide simulation results modal
 */
function hideResultsModal() {
  const modal = document.getElementById("results-modal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  }
}

/**
 * Fetch and display simulation results
 */
async function fetchAndDisplayResults() {
  try {
    const response = await fetch("/simulation/results");
    const results = await response.json();
    displayResults(results);
  } catch (error) {
    console.error("Failed to fetch results:", error);
  }
}

/**
 * Display simulation results in the modal
 */
function displayResults(results) {
  if (!results) return;

  // Update summary stats
  document.getElementById("results-duration").textContent =
    `${(results.duration_seconds || 0).toFixed(1)}s`;
  document.getElementById("results-steps").textContent = results.steps || 0;
  document.getElementById("results-avg-jn").textContent = (
    results.avg_Jn || 0
  ).toFixed(4);
  document.getElementById("results-avg-rn").textContent = (
    results.avg_rn || 0
  ).toFixed(2);
  document.getElementById("results-avg-path").textContent = (
    results.avg_traveled_path || 0
  ).toFixed(1);

  // Update final metrics
  document.getElementById("results-final-jn").textContent = (
    results.final_Jn || 0
  ).toFixed(4);
  document.getElementById("results-final-rn").textContent = (
    results.final_rn || 0
  ).toFixed(2);

  // Update status
  const statusEl = document.getElementById("results-status");
  if (results.completed) {
    if (results.destination_reached) {
      statusEl.innerHTML = `<span class="text-success font-medium">✓ Destination Reached</span>`;
      statusEl.className =
        "text-center py-2 rounded-lg bg-success/20 border border-success/30";
    } else {
      statusEl.innerHTML = `<span class="text-warning font-medium">Simulation Stopped</span>`;
      statusEl.className =
        "text-center py-2 rounded-lg bg-warning/20 border border-warning/30";
    }
  } else {
    statusEl.innerHTML = `<span class="text-muted-foreground">Simulation in progress...</span>`;
    statusEl.className = "text-center py-2 rounded-lg bg-secondary/30";
  }

  // Render chart
  renderResultsChart(results);
}

/**
 * Render results charts - two separate charts for Jn and rn
 */
function renderResultsChart(results) {
  const Jn = results.Jn_history || [];
  const rn = results.rn_history || [];

  // Calculate stats for Jn
  let jnMin = 0,
    jnMax = 1;
  if (Jn.length > 0) {
    const jnMean = Jn.reduce((a, b) => a + b, 0) / Jn.length;
    const jnStdDev = Math.sqrt(
      Jn.reduce((sum, x) => sum + Math.pow(x - jnMean, 2), 0) / Jn.length,
    );
    // Use mean ± 3 standard deviations for better visibility of fluctuations
    jnMin = Math.max(0, jnMean - 3 * jnStdDev);
    jnMax = Math.min(1, jnMean + 3 * jnStdDev);
    // Ensure some range if data is flat
    if (jnMax - jnMin < 0.01) {
      jnMin = Math.max(0, jnMean - 0.05);
      jnMax = Math.min(1, jnMean + 0.05);
    }
  }

  // Render Jn chart with dynamic range
  renderSingleChart(
    "results-chart-jn",
    Jn,
    "#22c55e",
    "Jn",
    jnMin,
    jnMax,
    4, // decimal places
  );

  // Render rn chart with auto-range
  const minRn = rn.length > 0 ? Math.min(...rn) * 0.9 : 0;
  const maxRn = rn.length > 0 ? Math.max(...rn) * 1.1 : 50;
  renderSingleChart(
    "results-chart-rn",
    rn,
    "#3b82f6",
    "rn",
    minRn,
    maxRn,
    1, // decimal places
  );
}

/**
 * Render a single metric chart
 */
function renderSingleChart(
  canvasId,
  data,
  color,
  label,
  minVal,
  maxVal,
  decimals,
) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  // Set canvas size with device pixel ratio for crisp rendering
  const container = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const rect = container.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = 150 * dpr;
  canvas.style.width = rect.width + "px";
  canvas.style.height = "150px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = 150;

  // Clear canvas with background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, width, height);

  if (!data || data.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No data", width / 2, height / 2);
    return;
  }

  const padding = { top: 15, right: 15, bottom: 25, left: 45 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calculate range
  const range = maxVal - minVal || 1;

  // Draw grid lines
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  // Draw axes
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Draw Y-axis labels
  ctx.fillStyle = "#888";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = maxVal - (i / 4) * range;
    const y = padding.top + (i / 4) * chartHeight;
    ctx.fillText(val.toFixed(decimals), padding.left - 5, y + 3);
  }

  // Downsample data if too many points
  const maxPoints = 300;
  const sampleRate = Math.max(1, Math.floor(data.length / maxPoints));

  // Draw filled area under line
  ctx.fillStyle = color + "20"; // 20% opacity
  ctx.beginPath();
  ctx.moveTo(padding.left, height - padding.bottom);
  for (let i = 0; i < data.length; i += sampleRate) {
    const x = padding.left + (i / (data.length - 1)) * chartWidth;
    const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  // Last point
  if ((data.length - 1) % sampleRate !== 0) {
    const x = padding.left + chartWidth;
    const y =
      padding.top +
      (1 - (data[data.length - 1] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(padding.left + chartWidth, height - padding.bottom);
  ctx.closePath();
  ctx.fill();

  // Draw line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < data.length; i += sampleRate) {
    const x = padding.left + (i / (data.length - 1)) * chartWidth;
    const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;
    if (first) {
      ctx.moveTo(x, y);
      first = false;
    } else ctx.lineTo(x, y);
  }
  // Last point
  if ((data.length - 1) % sampleRate !== 0) {
    const x = padding.left + chartWidth;
    const y =
      padding.top +
      (1 - (data[data.length - 1] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw current and average values
  if (data.length > 0) {
    const lastVal = data[data.length - 1];
    const avgVal = data.reduce((a, b) => a + b, 0) / data.length;

    ctx.fillStyle = color;
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(
      `Current: ${lastVal.toFixed(decimals)}`,
      width - padding.right,
      padding.top + 10,
    );

    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText(
      `Avg: ${avgVal.toFixed(decimals)}`,
      width - padding.right,
      padding.top + 22,
    );
  }

  // Draw X-axis label
  ctx.fillStyle = "#666";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${data.length} steps`, width / 2, height - 5);
}

/**
 * Render chart to a canvas at specified dimensions (for export)
 */
function renderChartToCanvas(
  canvas,
  ctx,
  width,
  height,
  data,
  color,
  label,
  minVal,
  maxVal,
  decimals,
) {
  // Clear canvas with background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, width, height);

  if (!data || data.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No data", width / 2, height / 2);
    return;
  }

  const padding = { top: 30, right: 30, bottom: 50, left: 80 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calculate range
  const range = maxVal - minVal || 1;

  // Draw title
  ctx.fillStyle = color;
  ctx.font = "bold 16px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, padding.left, 22);

  // Draw grid lines
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  // Draw axes
  ctx.strokeStyle = "#555";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Draw Y-axis labels
  ctx.fillStyle = "#aaa";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = maxVal - (i / 4) * range;
    const y = padding.top + (i / 4) * chartHeight;
    ctx.fillText(val.toFixed(decimals), padding.left - 10, y + 4);
  }

  // Downsample data if too many points
  const maxPoints = 500;
  const sampleRate = Math.max(1, Math.floor(data.length / maxPoints));

  // Draw filled area under line
  ctx.fillStyle = color + "30";
  ctx.beginPath();
  ctx.moveTo(padding.left, height - padding.bottom);
  for (let i = 0; i < data.length; i += sampleRate) {
    const x = padding.left + (i / (data.length - 1)) * chartWidth;
    const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  if ((data.length - 1) % sampleRate !== 0) {
    const x = padding.left + chartWidth;
    const y =
      padding.top +
      (1 - (data[data.length - 1] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(padding.left + chartWidth, height - padding.bottom);
  ctx.closePath();
  ctx.fill();

  // Draw line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < data.length; i += sampleRate) {
    const x = padding.left + (i / (data.length - 1)) * chartWidth;
    const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;
    if (first) {
      ctx.moveTo(x, y);
      first = false;
    } else ctx.lineTo(x, y);
  }
  if ((data.length - 1) % sampleRate !== 0) {
    const x = padding.left + chartWidth;
    const y =
      padding.top +
      (1 - (data[data.length - 1] - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw current and average values
  if (data.length > 0) {
    const lastVal = data[data.length - 1];
    const avgVal = data.reduce((a, b) => a + b, 0) / data.length;

    ctx.fillStyle = color;
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(
      `Final: ${lastVal.toFixed(decimals)}`,
      width - padding.right,
      padding.top + 15,
    );

    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText(
      `Avg: ${avgVal.toFixed(decimals)}`,
      width - padding.right,
      padding.top + 32,
    );
  }

  // Draw X-axis label
  ctx.fillStyle = "#888";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Time Steps (${data.length} total)`, width / 2, height - 15);
}

/**
 * Download charts as PNG (combines both charts at high resolution)
 */
async function downloadChartAsPNG() {
  // Fetch the actual data for re-rendering at high resolution
  let jnData = [];
  let rnData = [];

  try {
    const response = await fetch("/simulation/results");
    const results = await response.json();
    jnData = results.Jn_history || [];
    rnData = results.rn_history || [];
  } catch (error) {
    console.error("Failed to fetch results for export:", error);
    return;
  }

  if (jnData.length === 0 && rnData.length === 0) {
    alert("No data to export");
    return;
  }

  // Export dimensions (much larger for high quality)
  const exportWidth = 1200;
  const exportHeight = 400;
  const spacing = 40;
  const titleHeight = 60;

  // Create a combined canvas
  const combined = document.createElement("canvas");
  combined.width = exportWidth;
  combined.height = titleHeight + exportHeight * 2 + spacing * 2;

  const ctx = combined.getContext("2d");

  // Fill background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, combined.width, combined.height);

  // Draw main title
  ctx.fillStyle = "#fff";
  ctx.font = "bold 24px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Simulation Results", combined.width / 2, 38);

  // Draw timestamp
  ctx.fillStyle = "#666";
  ctx.font = "12px sans-serif";
  ctx.fillText(new Date().toLocaleString(), combined.width / 2, 55);

  // Calculate ranges for consistent scaling
  const jnMin = jnData.length > 0 ? Math.min(...jnData) * 0.95 : 0;
  const jnMax = jnData.length > 0 ? Math.max(...jnData) * 1.05 : 1;
  const rnMin = rnData.length > 0 ? Math.min(...rnData) * 0.95 : 0;
  const rnMax = rnData.length > 0 ? Math.max(...rnData) * 1.05 : 1;

  // Create temp canvases for each chart
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = exportWidth;
  tempCanvas.height = exportHeight;
  const tempCtx = tempCanvas.getContext("2d");

  // Render Jn chart
  renderChartToCanvas(
    tempCanvas,
    tempCtx,
    exportWidth,
    exportHeight,
    jnData,
    "#22c55e",
    "Jn (Average Communication Quality)",
    jnMin,
    jnMax,
    4,
  );
  ctx.drawImage(tempCanvas, 0, titleHeight);

  // Render Rn chart
  renderChartToCanvas(
    tempCanvas,
    tempCtx,
    exportWidth,
    exportHeight,
    rnData,
    "#3b82f6",
    "rn (Average Inter-Agent Distance)",
    rnMin,
    rnMax,
    2,
  );
  ctx.drawImage(tempCanvas, 0, titleHeight + exportHeight + spacing);

  // Download
  const link = document.createElement("a");
  link.download = `simulation_results_${Date.now()}.png`;
  link.href = combined.toDataURL("image/png");
  link.click();
}

/**
 * Download data as JSON
 */
async function downloadDataAsJSON() {
  try {
    const response = await fetch("/simulation/results/download?format=json");
    const data = await response.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    downloadBlob(blob, `simulation_data_${Date.now()}.json`);
  } catch (error) {
    console.error("Failed to download JSON:", error);
  }
}

/**
 * Download data as CSV
 */
async function downloadDataAsCSV() {
  try {
    const response = await fetch("/simulation/results/download?format=csv");
    const csv = await response.text();
    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `simulation_data_${Date.now()}.csv`);
  } catch (error) {
    console.error("Failed to download CSV:", error);
  }
}

/**
 * Helper to download a blob
 */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Check if simulation just completed and show results
 */
function checkSimulationCompletion(running) {
  // Only trigger if running is explicitly false (not undefined/null)
  if (lastSimulationRunning && running === false) {
    // Simulation just stopped - show results after a brief delay
    setTimeout(() => {
      // Double-check simulation is still stopped before showing modal
      // This prevents false triggers from brief API delays
      if (!App.simulationRunning) {
        showResultsModal();
      }
    }, 500);
  }
  // Only update lastSimulationRunning if we got a valid value
  if (running !== undefined && running !== null) {
    lastSimulationRunning = running;
  }
}

// ============================================================================
// PATH PLANNING MINI-MAP
// ============================================================================

const MiniMap = {
  canvas: null,
  canvasLarge: null,
  ctx: null,
  ctxLarge: null,
  visible: false,
  modalVisible: false,
  currentAlgorithm: "A*",

  // Grid configuration (should match backend)
  gridSize: 180, // Small canvas size (logical pixels)
  gridSizeLarge: 600, // Large canvas size (logical pixels)
  worldBounds: { x: [-50, 50], y: [-50, 150] }, // Will be updated from config

  // Stored data
  waypointsData: {}, // agent_id -> array of [x, y, z] waypoints
  discoveredObstaclesData: [], // Array of discovered obstacles from comm drop detection
  lastDrawData: null,

  // Device pixel ratio for high-DPI support
  dpr: 1,

  /**
   * Initialize the mini-map canvases with high-DPI support
   */
  init() {
    this.dpr = window.devicePixelRatio || 1;

    this.canvas = document.getElementById("minimap-canvas");
    this.canvasLarge = document.getElementById("minimap-canvas-large");

    // Setup small canvas with high-DPI
    if (this.canvas) {
      this.setupHighDPICanvas(this.canvas, this.gridSize);
      this.ctx = this.canvas.getContext("2d");
    }

    // Setup large canvas with high-DPI
    if (this.canvasLarge) {
      this.setupHighDPICanvas(this.canvasLarge, this.gridSizeLarge);
      this.ctxLarge = this.canvasLarge.getContext("2d");
    }

    // Re-init icons after DOM update
    if (window.lucide) {
      lucide.createIcons();
    }
  },

  /**
   * Setup a canvas for high-DPI rendering
   */
  setupHighDPICanvas(canvas, logicalSize) {
    // Set the actual size in memory (scaled for DPI)
    canvas.width = logicalSize * this.dpr;
    canvas.height = logicalSize * this.dpr;

    // Set the display size via CSS
    canvas.style.width = logicalSize + "px";
    canvas.style.height = logicalSize + "px";

    // Scale context to match DPI
    const ctx = canvas.getContext("2d");
    ctx.scale(this.dpr, this.dpr);
  },

  /**
   * Show/hide the mini-map based on path algorithm
   */
  setVisible(visible, algorithm = "A*") {
    const container = document.getElementById("pathplan-minimap");
    const algoLabel = document.getElementById("minimap-algo");
    const modalAlgoLabel = document.getElementById("minimap-modal-algo");
    if (container) {
      container.classList.toggle("hidden", !visible);
      this.visible = visible;
    }
    if (algoLabel) {
      algoLabel.textContent = algorithm;
    }
    if (modalAlgoLabel) {
      modalAlgoLabel.textContent = algorithm;
    }
    this.currentAlgorithm = algorithm;
  },

  /**
   * Show enlarged modal
   */
  showModal() {
    const modal = document.getElementById("minimap-modal");
    if (modal) {
      modal.classList.remove("hidden");
      this.modalVisible = true;

      // Re-setup large canvas for high-DPI (in case DPI changed)
      if (this.canvasLarge) {
        this.setupHighDPICanvas(this.canvasLarge, this.gridSizeLarge);
        this.ctxLarge = this.canvasLarge.getContext("2d");
      }

      // Re-init icons
      if (window.lucide) lucide.createIcons();

      // Draw on large canvas
      if (this.lastDrawData) {
        this.drawOnCanvas(this.ctxLarge, this.gridSizeLarge, this.lastDrawData);
      }
    }
  },

  /**
   * Hide modal
   */
  hideModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById("minimap-modal");
    if (modal) {
      modal.classList.add("hidden");
      this.modalVisible = false;
    }
  },

  /**
   * Store waypoints from visualization API
   */
  setWaypoints(waypointsDict) {
    this.waypointsData = waypointsDict || {};
    // Log when we receive waypoints for debugging
    const waypointCount = Object.keys(this.waypointsData).length;
    if (waypointCount > 0) {
      console.log(
        `[MiniMap] Received waypoints for ${waypointCount} agents:`,
        Object.entries(this.waypointsData).map(
          ([id, pts]) => `${id}: ${pts?.length || 0} points`,
        ),
      );
    }
  },

  /**
   * Set discovered obstacles for drawing
   */
  setDiscoveredObstacles(obstacles) {
    this.discoveredObstaclesData = obstacles || [];
    if (this.discoveredObstaclesData.length > 0) {
      console.log(
        `[MiniMap] Received ${this.discoveredObstaclesData.length} discovered obstacles`,
      );
    }
  },

  /**
   * Update world bounds from config
   */
  setBounds(bounds) {
    if (bounds) {
      this.worldBounds = bounds;
    }
  },

  /**
   * Convert world coordinates to canvas coordinates
   */
  worldToCanvas(x, y, canvasSize) {
    const { x: xBounds, y: yBounds } = this.worldBounds;
    const canvasX = ((x - xBounds[0]) / (xBounds[1] - xBounds[0])) * canvasSize;
    const canvasY =
      canvasSize - ((y - yBounds[0]) / (yBounds[1] - yBounds[0])) * canvasSize;
    return { x: canvasX, y: canvasY };
  },

  /**
   * Draw on a specific canvas context
   */
  drawOnCanvas(ctx, size, data) {
    if (!ctx) return;

    const scale = size / 180; // Scale factor relative to small canvas

    // Clear canvas
    ctx.fillStyle = "hsl(240, 10%, 8%)";
    ctx.fillRect(0, 0, size, size);

    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 0.5;
    const gridCells = 20;
    const cellSize = size / gridCells;
    for (let i = 0; i <= gridCells; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(size, i * cellSize);
      ctx.stroke();
    }

    // Draw actual jamming zones - color-coded by type
    // physical: Gray, low_jam: Yellow, high_jam: Red
    if (data.jammingZones) {
      const radiusScale =
        size / (this.worldBounds.x[1] - this.worldBounds.x[0]);

      // Color mapping for obstacle types
      const typeColors = {
        physical: { r: 128, g: 128, b: 128 }, // Gray
        low_jam: { r: 251, g: 191, b: 36 }, // Yellow
        high_jam: { r: 239, g: 68, b: 68 }, // Red
      };

      for (const zone of data.jammingZones) {
        if (!zone.active) continue;
        const pos = this.worldToCanvas(zone.center[0], zone.center[1], size);
        const physicalRadius = zone.radius * radiusScale;
        const jammingFieldRadius =
          (zone.jamming_radius || zone.radius * 2) * radiusScale;

        // Get color based on obstacle type (default to low_jam for backward compat)
        const obstacleType = zone.obstacle_type || "low_jam";
        const color = typeColors[obstacleType] || typeColors.low_jam;
        const { r, g, b } = color;

        // For physical obstacles, only show inner circle (no jamming field)
        const isPhysical = obstacleType === "physical";

        if (!isPhysical) {
          // OUTER circle - Jamming field (where comm degradation starts)
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.1)`;
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.4)`;
          ctx.lineWidth = 1 * scale;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, jammingFieldRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // INNER circle - Physical obstacle (solid core)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${isPhysical ? 0.5 : 0.4})`;
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        ctx.lineWidth = 1.5 * scale;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, physicalRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Draw DISCOVERED obstacle points (detected via comm drop) as simple dots
    // These mark where vehicles detected jamming through comm quality drop
    if (data.discoveredObstacles && data.discoveredObstacles.length > 0) {
      ctx.fillStyle = "rgba(251, 191, 36, 1)"; // Yellow/gold for discovered points
      for (const obs of data.discoveredObstacles) {
        const pos = this.worldToCanvas(obs.center[0], obs.center[1], size);
        // Draw as a simple dot
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 3 * scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw waypoints - each agent gets its own path line with semi-transparent color
    if (data.waypoints && Object.keys(data.waypoints).length > 0) {
      // Semi-transparent colors for individual agent paths
      const colors = [
        "rgba(6, 182, 212, 0.6)", // cyan
        "rgba(168, 85, 247, 0.6)", // purple
        "rgba(34, 197, 94, 0.6)", // green
        "rgba(251, 146, 60, 0.6)", // orange
        "rgba(236, 72, 153, 0.6)", // pink
        "rgba(251, 191, 36, 0.6)", // yellow
        "rgba(129, 140, 248, 0.6)", // indigo
      ];

      let colorIndex = 0;
      for (const [agentId, waypoints] of Object.entries(data.waypoints)) {
        if (!waypoints || waypoints.length < 2) continue;

        const color = colors[colorIndex % colors.length];
        colorIndex++;

        // Draw path line with thicker stroke for visibility
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * scale;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        for (let i = 0; i < waypoints.length; i++) {
          const wp = this.worldToCanvas(waypoints[i][0], waypoints[i][1], size);
          if (i === 0) {
            ctx.moveTo(wp.x, wp.y);
          } else {
            ctx.lineTo(wp.x, wp.y);
          }
        }
        ctx.stroke();

        // Draw waypoint dots at key positions
        ctx.fillStyle = color.replace("0.6)", "0.9)"); // More opaque for dots
        for (let i = 0; i < waypoints.length; i++) {
          // Draw start, end, and every 5th point
          if (i !== 0 && i !== waypoints.length - 1 && i % 5 !== 0) continue;
          const wp = this.worldToCanvas(waypoints[i][0], waypoints[i][1], size);
          ctx.beginPath();
          ctx.arc(wp.x, wp.y, 2 * scale, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Draw destination (goal) as blue ring with dot (like 3D sim)
    if (data.destination) {
      const dest = this.worldToCanvas(
        data.destination[0],
        data.destination[1],
        size,
      );
      const ringRadius = 6 * scale;
      const dotRadius = 2 * scale;

      // Outer ring
      ctx.strokeStyle = "rgba(79, 195, 247, 0.9)"; // Light blue (mission color)
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      ctx.arc(dest.x, dest.y, ringRadius, 0, Math.PI * 2);
      ctx.stroke();

      // Inner dot
      ctx.fillStyle = "rgba(79, 195, 247, 1)";
      ctx.beginPath();
      ctx.arc(dest.x, dest.y, dotRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw agents as green triangles
    if (data.agents) {
      ctx.fillStyle = "rgba(34, 197, 94, 0.95)";
      ctx.strokeStyle = "rgba(34, 197, 94, 1)";
      ctx.lineWidth = 0.5 * scale;
      for (const agent of data.agents) {
        const pos = this.worldToCanvas(
          agent.position[0],
          agent.position[1],
          size,
        );
        const heading = agent.heading || 0;
        const triSize = 4 * scale;

        ctx.save();
        ctx.translate(pos.x, pos.y);
        ctx.rotate(-heading + Math.PI / 2);
        ctx.beginPath();
        ctx.moveTo(0, -triSize);
        ctx.lineTo(-triSize * 0.7, triSize * 0.8);
        ctx.lineTo(triSize * 0.7, triSize * 0.8);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }
  },

  /**
   * Draw the mini-map with current state
   */
  draw(data) {
    if (!this.visible) return;

    // Include stored waypoints
    data.waypoints = this.waypointsData;
    // Include stored discovered obstacles
    data.discoveredObstacles = this.discoveredObstaclesData || [];
    this.lastDrawData = data;

    // Draw on small canvas
    this.drawOnCanvas(this.ctx, this.gridSize, data);

    // Also draw on large canvas if modal is open
    if (this.modalVisible) {
      this.drawOnCanvas(this.ctxLarge, this.gridSizeLarge, data);
    }
  },
};

/**
 * Update mini-map from current app state
 */
function updateMiniMap() {
  // Get current path algorithm
  const pathAlgoSelect = document.getElementById("path-algo-select");
  const pathAlgo = pathAlgoSelect ? pathAlgoSelect.value : "direct";

  // Show mini-map only for grid-based algorithms
  const gridBasedAlgos = ["astar", "rrt", "theta_star", "bi_astar"];
  const showMiniMap =
    gridBasedAlgos.includes(pathAlgo) && App.simulationRunning;

  const algoNames = {
    astar: "A*",
    rrt: "RRT",
    theta_star: "Theta*",
    bi_astar: "Bi-A*",
  };
  MiniMap.setVisible(
    showMiniMap,
    algoNames[pathAlgo] || pathAlgo.toUpperCase(),
  );

  if (!showMiniMap) return;

  // Collect data for mini-map
  const data = {
    agents: [],
    jammingZones: [],
    destination: null,
    waypoints: {},
  };

  // Get data from Scene3D
  if (window.Scene3D) {
    const config = Scene3D.getConfig ? Scene3D.getConfig() : null;
    if (config && config.bounds) {
      MiniMap.setBounds(config.bounds);
    }
    if (config && config.missionEnd) {
      data.destination = config.missionEnd;
    }

    // Get vehicle states directly from Scene3D
    if (Scene3D.getAllVehicleStates) {
      data.agents = Scene3D.getAllVehicleStates();
    }

    // Get jamming zones directly from Scene3D
    if (Scene3D.getAllJammingZones) {
      data.jammingZones = Scene3D.getAllJammingZones();
    }
  }

  MiniMap.draw(data);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  App.init();
  MiniMap.init();

  // Ensure all lucide icons are created (including new minimap icons)
  if (window.lucide) {
    lucide.createIcons();
  }

  // Visualization toggle event listeners
  const trailsToggle = document.getElementById("toggle-trails");

  if (trailsToggle) {
    trailsToggle.addEventListener("change", (e) => {
      if (window.Scene3D) {
        Scene3D.setTrailsVisible(e.target.checked);
      }
    });
  }

  // Trail length dropdown
  const trailLengthSelect = document.getElementById("trail-length");
  if (trailLengthSelect) {
    trailLengthSelect.addEventListener("change", (e) => {
      App.trailLength = e.target.value;
      console.log(`[App] Trail length set to: ${App.trailLength}`);
    });
  }

  // Update mini-map periodically when simulation is running
  setInterval(() => {
    if (App.simulationRunning) {
      updateMiniMap();
    }
  }, 500);
});

window.App = App;
window.MiniMap = MiniMap;
