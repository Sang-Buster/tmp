/**
 * 3D Scene with Three.js for vehicle simulation visualization
 * Enhanced with labeled axes, tooltips, jamming zones
 */

/**
 * Format ID for display (agent1 -> Agent 1, obstacle_1 -> Obstacle 1)
 */
function formatDisplayName(id) {
  if (!id) return id;
  return id
    .replace(/_/g, " ")
    .replace(/([a-z])(\d)/gi, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Scene objects
let scene, camera, renderer, controls;
let vehicles = {};
let jammingZones = {};
let spoofingZones = {};
let phantomVehicles = {};
let falsificationLines = {};
let gridHelper, missionMarker;

// Axis indicator (corner orientation gizmo)
let axisScene, axisCamera, axisRenderer;

// Visualization objects
let communicationLinks = []; // Array of Line objects
let waypointPaths = {}; // agent_id -> Line object
let waypointMarkers = {}; // agent_id -> Array of sphere markers
let llmGuidanceArrows = {}; // agent_id -> Arrow helper for LLM guidance
let avoidanceVectors = {}; // agent_id -> Arrow helper for avoidance
let llmTargetLines = {}; // agent_id -> Line for chat-commanded targets
let traveledPaths = {}; // agent_id -> Line object for traveled trail

// Visibility toggles
let showWaypoints = true;
let showTrails = true;

// Interaction
let raycaster, mouse;
let selectedVehicle = null;
let hoveredVehicle = null;
let selectedJammingZone = null;
let selectedSpoofingZone = null;
let tooltip = null;

// Configuration - will be updated from API
let CONFIG = {
  bounds: { x: [-200, 200], y: [-200, 200], z: [0, 200] },
  missionEnd: [35, 150, 30],
  colors: {
    clear: 0x4caf50,
    jammed: 0xf44336,
    target: 0xffeb3b,
    grid: 0x2a2a4a,
    mission: 0x4fc3f7,
    jamming: 0xff4444,
    selected: 0x00ffff,
    commLink: 0x88ff88, // Communication links (green)
    commLinkWeak: 0xffff88, // Weak links (yellow)
    waypoint: 0x88ccff, // Waypoint path (cyan)
    waypointMarker: 0xff8844, // Waypoint markers (orange)
    llmGuidance: 0xc084fc, // LLM guidance arrows (purple)
    avoidance: 0xf87171, // Avoidance vectors (red)
    trail: 0xfbbf24, // Traveled path trail (amber/yellow)
    // Obstacle type colors
    obstacleType: {
      physical: 0x808080, // Gray - hard obstacle
      low_jam: 0xfbbf24, // Yellow - low-power jamming
      high_jam: 0xef4444, // Red - high-power jamming
    },
    axis: {
      x: 0xff6666,
      y: 0x66ff66,
      z: 0x6666ff,
    },
  },
};

/**
 * Initialize Three.js scene
 */
async function initScene(container) {
  // Fetch config from API first
  await fetchConfig();

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d15);
  scene.fog = new THREE.Fog(0x0d0d15, 150, 400);

  // Camera - position based on scene size
  const aspect = container.clientWidth / container.clientHeight;
  camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 2000);

  // Position camera to see the whole scene
  const centerY = (CONFIG.bounds.y[0] + CONFIG.bounds.y[1]) / 2;
  const rangeY = CONFIG.bounds.y[1] - CONFIG.bounds.y[0];
  camera.position.set(80, 60, centerY);
  camera.lookAt(0, 0, centerY);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.minDistance = 20;
  controls.maxDistance = 500;
  controls.target.set(0, 0, centerY);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(50, 100, 50);
  scene.add(directionalLight);

  // Raycaster for interactions
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // Create scene elements
  createGrid();
  createMissionMarker();
  createTooltip(container);

  // Create axis orientation indicator in corner
  createAxisIndicator();

  // Event listeners
  renderer.domElement.addEventListener("mousemove", onMouseMove);
  renderer.domElement.addEventListener("click", onMouseClick);

  window.addEventListener("resize", () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });

  // Start animation loop
  animate();

  console.log("[Scene3D] Initialized with config:", CONFIG);
}

/**
 * Fetch configuration from API
 */
async function fetchConfig() {
  try {
    const response = await fetch("/status");
    const data = await response.json();

    if (data.boundaries) {
      CONFIG.bounds = {
        x: data.boundaries.x_range,
        y: data.boundaries.y_range,
        z: data.boundaries.z_range,
      };
      CONFIG.missionEnd = data.boundaries.mission_end;
      console.log("[Scene3D] Config loaded:", CONFIG);
    }
  } catch (error) {
    console.warn("[Scene3D] Failed to fetch config, using defaults:", error);
  }
}

/**
 * Create ground grid based on bounds
 */
function createGrid() {
  // Calculate grid size based on bounds
  const xRange = CONFIG.bounds.x[1] - CONFIG.bounds.x[0];
  const yRange = CONFIG.bounds.y[1] - CONFIG.bounds.y[0];
  const size = Math.max(xRange, yRange);
  const divisions = Math.floor(size / 20); // Grid lines every 20 units

  // Center the grid
  const centerX = (CONFIG.bounds.x[0] + CONFIG.bounds.x[1]) / 2;
  const centerY = (CONFIG.bounds.y[0] + CONFIG.bounds.y[1]) / 2;

  gridHelper = new THREE.GridHelper(
    size,
    divisions,
    CONFIG.colors.grid,
    CONFIG.colors.grid,
  );
  gridHelper.position.set(centerX, 0, centerY);
  scene.add(gridHelper);

  // Ground plane
  const groundGeometry = new THREE.PlaneGeometry(xRange * 1.2, yRange * 1.2);
  const groundMaterial = new THREE.MeshStandardMaterial({
    color: 0x0a0a10,
    transparent: true,
    opacity: 0.5,
  });
  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(centerX, -0.01, centerY);
  ground.name = "ground";
  scene.add(ground);

  // Add subtle colored axis lines on ground for orientation
  const axisMaterial = {
    x: new THREE.LineBasicMaterial({
      color: CONFIG.colors.axis.x,
      transparent: true,
      opacity: 0.4,
    }),
    y: new THREE.LineBasicMaterial({
      color: CONFIG.colors.axis.y,
      transparent: true,
      opacity: 0.4,
    }),
  };

  // X axis line (red, along sim X)
  const xPoints = [
    new THREE.Vector3(CONFIG.bounds.x[0], 0.1, 0),
    new THREE.Vector3(CONFIG.bounds.x[1], 0.1, 0),
  ];
  const xLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(xPoints),
    axisMaterial.x,
  );
  scene.add(xLine);

  // Y axis line (green, along sim Y which is Three.js Z)
  const yPoints = [
    new THREE.Vector3(0, 0.1, CONFIG.bounds.y[0]),
    new THREE.Vector3(0, 0.1, CONFIG.bounds.y[1]),
  ];
  const yLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(yPoints),
    axisMaterial.y,
  );
  scene.add(yLine);
}

/**
 * Create axis orientation indicator in corner
 * This is a small 3D gizmo that shows X, Y, Z orientation synced with main camera
 */
function createAxisIndicator() {
  const container = document.getElementById("axis-indicator");
  if (!container) return;

  // Create mini scene for axis indicator
  axisScene = new THREE.Scene();

  // Orthographic camera for consistent size
  axisCamera = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 10);
  axisCamera.position.set(0, 0, 4);
  axisCamera.lookAt(0, 0, 0);

  // Mini renderer
  axisRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  axisRenderer.setSize(90, 90);
  axisRenderer.setClearColor(0x000000, 0);
  container.appendChild(axisRenderer.domElement);

  // Create axis arrows with labels
  const axisLength = 1.2;
  const arrowRadius = 0.06;
  const arrowLength = 0.25;

  // X Axis (Red) - negated to match main scene (sim +X is right)
  createAxisArrow(
    axisScene,
    [-1, 0, 0],
    CONFIG.colors.axis.x,
    axisLength,
    arrowRadius,
    arrowLength,
    "X",
  );

  // Y Axis (Green) - our sim Y maps to Three.js Z
  createAxisArrow(
    axisScene,
    [0, 0, 1],
    CONFIG.colors.axis.y,
    axisLength,
    arrowRadius,
    arrowLength,
    "Y",
  );

  // Z Axis (Blue) - our sim Z maps to Three.js Y (up)
  createAxisArrow(
    axisScene,
    [0, 1, 0],
    CONFIG.colors.axis.z,
    axisLength,
    arrowRadius,
    arrowLength,
    "Z",
  );

  // Add ambient light
  axisScene.add(new THREE.AmbientLight(0xffffff, 0.8));

  console.log("[Scene3D] Axis indicator created");
}

function createAxisArrow(
  targetScene,
  direction,
  color,
  length,
  radius,
  arrowLen,
  label,
) {
  const [dx, dy, dz] = direction;

  // Cylinder for the shaft
  const shaftGeom = new THREE.CylinderGeometry(
    radius * 0.5,
    radius * 0.5,
    length - arrowLen,
    8,
  );
  const shaftMat = new THREE.MeshStandardMaterial({
    color: color,
    emissive: color,
    emissiveIntensity: 0.3,
  });
  const shaft = new THREE.Mesh(shaftGeom, shaftMat);

  // Position and rotate shaft
  shaft.position.set(
    (dx * (length - arrowLen)) / 2,
    (dy * (length - arrowLen)) / 2,
    (dz * (length - arrowLen)) / 2,
  );

  // Rotate to align with axis
  if (dx !== 0) shaft.rotation.z = Math.PI / 2;
  else if (dz !== 0) shaft.rotation.x = -Math.PI / 2;

  targetScene.add(shaft);

  // Cone for the arrow head
  const coneGeom = new THREE.ConeGeometry(radius * 2, arrowLen, 8);
  const coneMat = new THREE.MeshStandardMaterial({
    color: color,
    emissive: color,
    emissiveIntensity: 0.3,
  });
  const cone = new THREE.Mesh(coneGeom, coneMat);

  cone.position.set(
    dx * (length - arrowLen / 2),
    dy * (length - arrowLen / 2),
    dz * (length - arrowLen / 2),
  );

  // Rotate cone to point along axis
  if (dx !== 0) cone.rotation.z = -Math.PI / 2;
  else if (dz !== 0) cone.rotation.x = Math.PI / 2;

  targetScene.add(cone);

  // Add text label near arrow tip
  if (label) {
    const labelSprite = createAxisLabel(label, color);
    labelSprite.position.set(
      dx * (length + 0.3),
      dy * (length + 0.3),
      dz * (length + 0.3),
    );
    targetScene.add(labelSprite);
  }
}

function createAxisLabel(text, color) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 64;
  canvas.height = 64;

  ctx.fillStyle = "transparent";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.font = "bold 48px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#" + color.toString(16).padStart(6, "0");
  ctx.fillText(text, 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.6, 0.6, 1);

  return sprite;
}

/**
 * Create text sprite for labels
 */
function createTextSprite(text, color, fontSize = 16) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 128;
  canvas.height = 64;

  ctx.fillStyle = "transparent";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.font = `bold ${fontSize}px Arial`;
  ctx.fillStyle = "#" + color.toString(16).padStart(6, "0");
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(2, 1, 1);

  return sprite;
}

/**
 * Create mission endpoint marker (ring only, hover for details)
 */
function createMissionMarker() {
  const [mx, my, mz] = CONFIG.missionEnd;

  // Ring at destination height - smaller size
  const ringGeometry = new THREE.TorusGeometry(3, 0.3, 16, 48);
  const ringMaterial = new THREE.MeshStandardMaterial({
    color: CONFIG.colors.mission,
    emissive: CONFIG.colors.mission,
    emissiveIntensity: 0.4,
  });
  missionMarker = new THREE.Mesh(ringGeometry, ringMaterial);
  missionMarker.rotation.x = Math.PI / 2;
  missionMarker.position.set(-mx, mz, my); // -x (negated), z(up), y
  missionMarker.name = "destination";
  missionMarker.userData = {
    type: "destination",
    position: { x: mx, y: my, z: mz },
  };
  scene.add(missionMarker);

  // Add a subtle vertical line from ground to ring for visibility (negate X)
  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-mx, 0, my),
    new THREE.Vector3(-mx, mz, my),
  ]);
  const lineMaterial = new THREE.LineDashedMaterial({
    color: CONFIG.colors.mission,
    dashSize: 3,
    gapSize: 2,
    transparent: true,
    opacity: 0.3,
  });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  line.computeLineDistances();
  scene.add(line);

  console.log(`[Scene3D] Mission marker at (${mx}, ${my}, ${mz})`);
}

/**
 * Create tooltip element
 */
function createTooltip(container) {
  tooltip = document.createElement("div");
  tooltip.id = "vehicle-tooltip";
  tooltip.style.cssText = `
        position: absolute;
        background: hsl(240 10% 6% / 0.95);
        border: 1px solid hsl(240 3.7% 20%);
        border-radius: 8px;
        padding: 12px 16px;
        color: hsl(0 0% 98%);
        font-size: 12px;
        pointer-events: none;
        display: none;
        z-index: 1000;
        max-width: 260px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        backdrop-filter: blur(8px);
    `;
  container.style.position = "relative";
  container.appendChild(tooltip);
}

/**
 * Create or update a vehicle in the scene
 */
function updateVehicle(agentId, data) {
  const pos = data.position || [0, 0, 0];
  const jammed = data.jammed || false;
  const target = data.llm_target;
  const heading = data.heading || 0;
  const isPhantom = data.is_phantom || false;

  // Don't create 3D objects for phantom agents here -- they use phantomVehicles
  if (isPhantom) return;

  if (!vehicles[agentId]) {
    // Create new vehicle
    const group = new THREE.Group();
    group.userData = { agentId, data: {} };

    // Body (box)
    const bodyGeometry = new THREE.BoxGeometry(0.8, 0.4, 1.2);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: jammed ? CONFIG.colors.jammed : CONFIG.colors.clear,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.3;
    body.name = "body";
    group.add(body);

    // Top (smaller box)
    const topGeometry = new THREE.BoxGeometry(0.6, 0.3, 0.6);
    const topMaterial = new THREE.MeshStandardMaterial({ color: 0x333344 });
    const top = new THREE.Mesh(topGeometry, topMaterial);
    top.position.y = 0.65;
    top.position.z = -0.1;
    group.add(top);

    // Antenna
    const antennaGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.4, 8);
    const antennaMaterial = new THREE.MeshStandardMaterial({ color: 0x888888 });
    const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
    antenna.position.set(0, 1, 0);
    group.add(antenna);

    // Status light
    const lightGeometry = new THREE.SphereGeometry(0.12, 8, 8);
    const lightMaterial = new THREE.MeshStandardMaterial({
      color: jammed ? CONFIG.colors.jammed : CONFIG.colors.clear,
      emissive: jammed ? CONFIG.colors.jammed : CONFIG.colors.clear,
      emissiveIntensity: 0.5,
    });
    const light = new THREE.Mesh(lightGeometry, lightMaterial);
    light.position.set(0, 1.2, 0);
    light.name = "statusLight";
    group.add(light);

    // Selection ring (hidden by default)
    const selectRingGeometry = new THREE.TorusGeometry(1, 0.05, 8, 32);
    const selectRingMaterial = new THREE.MeshStandardMaterial({
      color: CONFIG.colors.selected,
      emissive: CONFIG.colors.selected,
      emissiveIntensity: 0.5,
    });
    const selectRing = new THREE.Mesh(selectRingGeometry, selectRingMaterial);
    selectRing.rotation.x = Math.PI / 2;
    selectRing.position.y = 0.1;
    selectRing.visible = false;
    selectRing.name = "selectRing";
    group.add(selectRing);

    // Target line (initially hidden)
    const lineMaterial = new THREE.LineDashedMaterial({
      color: CONFIG.colors.target,
      dashSize: 0.3,
      gapSize: 0.2,
    });
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setFromPoints([
      new THREE.Vector3(0, 0.5, 0),
      new THREE.Vector3(0, 0.5, 0),
    ]);
    const targetLine = new THREE.Line(lineGeometry, lineMaterial);
    targetLine.computeLineDistances();
    targetLine.visible = false;
    targetLine.name = "targetLine";
    group.add(targetLine);

    // Agent label
    const labelSprite = createTextSprite(agentId, 0xffffff, 12);
    labelSprite.position.set(0, 2, 0);
    labelSprite.name = "label";
    group.add(labelSprite);

    group.name = agentId;
    scene.add(group);
    vehicles[agentId] = group;
  }

  // Store full data
  const vehicle = vehicles[agentId];
  vehicle.userData.data = data;

  // Update position (swap Y and Z for Three.js coordinate system, negate X)
  // pos is [x, y, z] in sim coords -> Three.js (-x, z, y)
  // X is negated so negative sim X appears on left side of scene
  vehicle.position.set(-pos[0], pos[2] || 0, pos[1]);

  // Update rotation based on heading
  // heading is arctan2(vel_y, vel_x) in sim coords
  // In Three.js, rotation.y rotates around the vertical axis
  // Since we negated X, adjust heading: heading + π/2 instead of -heading + π/2
  vehicle.rotation.y = heading + Math.PI / 2;

  // Update colors based on jammed status
  const body = vehicle.getObjectByName("body");
  const statusLight = vehicle.getObjectByName("statusLight");
  const color = jammed ? CONFIG.colors.jammed : CONFIG.colors.clear;

  try {
    if (body && body.material) body.material.color.setHex(color);
    if (statusLight && statusLight.material) {
      statusLight.material.color.setHex(color);
      if (statusLight.material.emissive) {
        statusLight.material.emissive.setHex(color);
      }
    }
  } catch (e) {
    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "scene3d.js:updateVehicle:colorError",
        message: "Vehicle color update error",
        data: {
          agentId: agentId,
          jammed: jammed,
          error: e.message,
          bodyMaterial: !!body?.material,
          lightMaterial: !!statusLight?.material,
        },
        timestamp: Date.now(),
        sessionId: "debug-session",
        hypothesisId: "H1",
      }),
    }).catch(() => {});
    // #endregion
  }

  // Note: Yellow target line removed - using cyan updateLLMTargetLines() instead
  // The cyan dashed line from updateLLMTargetLines provides clearer visualization
  const targetLine = vehicle.getObjectByName("targetLine");
  if (targetLine) {
    targetLine.visible = false;
  }

  // Update selection ring
  const selectRing = vehicle.getObjectByName("selectRing");
  if (selectRing) {
    selectRing.visible = selectedVehicle === agentId;
  }

  // Update tooltip if this vehicle is selected
  if (
    selectedVehicle === agentId &&
    tooltip &&
    tooltip.style.display !== "none"
  ) {
    updateSelectedVehicleTooltip(data);
  }
}

/**
 * Update tooltip content for selected vehicle (called during updates)
 */
function updateSelectedVehicleTooltip(data) {
  if (!tooltip || !data) return;

  const pos = data.position || [0, 0, 0];
  const vel = data.velocity || [0, 0, 0];
  const speed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2);
  const statusColor = data.jammed ? "#ef4444" : "#22c55e";
  const statusText = data.jammed ? "JAMMED" : "OK";

  tooltip.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
      <span style="font-weight: 600; font-size: 13px;">${formatDisplayName(data.agent_id) || "Unknown"}</span>
      <span style="font-size: 10px; padding: 2px 6px; border-radius: 4px; background: ${statusColor}20; color: ${statusColor};">${statusText}</span>
    </div>
    <div style="display: grid; grid-template-columns: auto auto; gap: 4px 16px; font-size: 11px;">
      <span style="color: hsl(240 5% 64.9%);">Position</span>
      <span>(${pos[0]?.toFixed(1)}, ${pos[1]?.toFixed(1)}, ${pos[2]?.toFixed(1)})</span>
      <span style="color: hsl(240 5% 64.9%);">Speed</span>
      <span>${speed.toFixed(2)} m/s</span>
      <span style="color: hsl(240 5% 64.9%);">Heading</span>
      <span>${(data.heading_degrees || 0).toFixed(0)}°</span>
      <span style="color: hsl(240 5% 64.9%);">Comm</span>
      <span>${((data.communication_quality || 0) * 100).toFixed(0)}%</span>
      ${data.distance_to_goal ? `<span style="color: hsl(240 5% 64.9%);">To Goal</span><span>${data.distance_to_goal.toFixed(1)} m</span>` : ""}
    </div>
    ${
      data.llm_target
        ? `
      <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid hsl(240 3.7% 20%); font-size: 10px; color: #facc15;">
        Target: (${data.llm_target[0]?.toFixed(1)}, ${data.llm_target[1]?.toFixed(1)}, ${(data.llm_target[2] || 0).toFixed(1)})
      </div>
    `
        : ""
    }
  `;
}

/**
 * Remove a vehicle from the scene
 */
function removeVehicle(agentId) {
  if (vehicles[agentId]) {
    scene.remove(vehicles[agentId]);
    delete vehicles[agentId];
  }
}

/**
 * Update all vehicles from API data
 */
function updateAllVehicles(agentsData) {
  // Null safety - if no data, don't crash and keep existing vehicles
  if (!agentsData || typeof agentsData !== "object") {
    console.warn(
      "[Scene3D] updateAllVehicles called with invalid data - keeping existing vehicles",
    );
    return;
  }

  // If agentsData is empty, it's likely a timing issue - don't remove existing vehicles
  const agentCount = Object.keys(agentsData).length;
  if (agentCount === 0 && Object.keys(vehicles).length > 0) {
    // API returned empty data but we have vehicles - keep them (timing issue)
    return;
  }

  for (const [agentId, data] of Object.entries(agentsData)) {
    if (data) {
      updateVehicle(agentId, data);
    }
  }

  // Only remove vehicles that are explicitly not in the new data
  // (and only if we received actual agent data)
  if (agentCount > 0) {
    for (const agentId of Object.keys(vehicles)) {
      if (!agentsData[agentId]) {
        removeVehicle(agentId);
      }
    }
  }
}

/**
 * Create or update a jamming zone
 */
function updateJammingZone(zoneId, zoneData) {
  const center = zoneData.center || [0, 0, 0];
  const radius = zoneData.radius || 5; // Physical radius r_obs
  const jammingRadius = zoneData.jamming_radius || radius * 2; // Jamming field r_jam = κ_J × r_obs
  const active = zoneData.active !== false;
  const obstacleType = zoneData.obstacle_type || "low_jam"; // Default to low_jam for backward compat

  // Get color based on obstacle type
  const typeColor =
    CONFIG.colors.obstacleType[obstacleType] || CONFIG.colors.jamming;

  // For physical obstacles, don't show outer sphere (no jamming field)
  const isPhysical = obstacleType === "physical";
  const showOuterSphere = !isPhysical;

  if (!jammingZones[zoneId]) {
    // Create new jamming zone
    const group = new THREE.Group();

    // OUTER sphere - Jamming field boundary (only for jamming types, not physical)
    if (showOuterSphere) {
      const outerGeometry = new THREE.SphereGeometry(jammingRadius, 32, 32);
      const outerMaterial = new THREE.MeshBasicMaterial({
        color: typeColor,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: true,
      });
      const outerSphere = new THREE.Mesh(outerGeometry, outerMaterial);
      outerSphere.name = "outerSphere";
      group.add(outerSphere);

      // Outer wireframe - shows jamming field boundary
      const outerWireGeometry = new THREE.SphereGeometry(jammingRadius, 24, 24);
      const outerWireMaterial = new THREE.MeshBasicMaterial({
        color: typeColor,
        wireframe: true,
        transparent: true,
        opacity: 0.3,
        depthTest: true,
        depthWrite: false,
      });
      const outerWireframe = new THREE.Mesh(
        outerWireGeometry,
        outerWireMaterial,
      );
      outerWireframe.name = "outerWireframe";
      group.add(outerWireframe);
    }

    // INNER sphere - Physical obstacle (solid core)
    const innerGeometry = new THREE.SphereGeometry(radius, 24, 24);
    const innerMaterial = new THREE.MeshBasicMaterial({
      color: typeColor,
      transparent: true,
      opacity: isPhysical ? 0.6 : 0.5, // Physical obstacles slightly more opaque
      side: THREE.DoubleSide,
      depthWrite: true,
      depthTest: true,
    });
    const innerSphere = new THREE.Mesh(innerGeometry, innerMaterial);
    innerSphere.name = "sphere";
    group.add(innerSphere);

    // Inner wireframe - physical boundary
    const innerWireGeometry = new THREE.SphereGeometry(radius, 16, 16);
    const innerWireMaterial = new THREE.MeshBasicMaterial({
      color: typeColor,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
      depthTest: true,
      depthWrite: false,
    });
    const innerWireframe = new THREE.Mesh(innerWireGeometry, innerWireMaterial);
    innerWireframe.name = "wireframe";
    group.add(innerWireframe);

    // Center marker
    const centerGeometry = new THREE.SphereGeometry(0.3, 8, 8);
    const centerMaterial = new THREE.MeshBasicMaterial({
      color: typeColor,
    });
    const centerMarker = new THREE.Mesh(centerGeometry, centerMaterial);
    centerMarker.name = "center";
    group.add(centerMarker);

    // Label - show type and radius info
    const typeLabel =
      obstacleType === "physical"
        ? "PHYS"
        : obstacleType === "high_jam"
          ? "HIGH"
          : "LOW";
    const labelText = isPhysical
      ? `${typeLabel} r=${radius}`
      : `${typeLabel} r=${radius} field=${jammingRadius}`;
    const label = createTextSprite(labelText, typeColor, 12);
    label.position.set(0, (isPhysical ? radius : jammingRadius) + 1, 0);
    label.name = "label";
    group.add(label);

    group.name = zoneId;
    group.userData = { zoneId };
    scene.add(group);
    jammingZones[zoneId] = group;

    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "scene3d.js:updateJammingZone:created",
        message: "Jamming zone CREATED",
        data: {
          zoneId: zoneId,
          center: center,
          radius: radius,
          active: active,
        },
        timestamp: Date.now(),
        sessionId: "debug-session",
        hypothesisId: "C-render",
      }),
    }).catch(() => {});
    // #endregion
  }

  const zone = jammingZones[zoneId];
  const oldVisible = zone.visible;
  // Negate X so negative sim X appears on left side of scene
  zone.position.set(-center[0], center[2] || 0, center[1]);
  zone.visible = active;

  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "scene3d.js:updateJammingZone:updated",
      message: "Jamming zone updated",
      data: {
        zoneId: zoneId,
        position: zone.position.toArray(),
        visible: zone.visible,
        active: active,
        visibilityChanged: oldVisible !== zone.visible,
        inScene: scene.children.includes(zone),
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      hypothesisId: "C-render",
    }),
  }).catch(() => {});
  // #endregion

  // Update radius if changed
  const sphere = zone.getObjectByName("sphere");
  const wireframe = zone.getObjectByName("wireframe");
  if (
    sphere &&
    wireframe &&
    sphere.geometry.parameters &&
    sphere.geometry.parameters.radius !== radius
  ) {
    sphere.geometry.dispose();
    sphere.geometry = new THREE.SphereGeometry(radius, 24, 24);
    wireframe.geometry.dispose();
    wireframe.geometry = new THREE.SphereGeometry(radius, 16, 16);
  }
}

/**
 * Remove a jamming zone
 */
function removeJammingZone(zoneId) {
  if (jammingZones[zoneId]) {
    scene.remove(jammingZones[zoneId]);
    delete jammingZones[zoneId];
  }
}

/**
 * Update all jamming zones
 */
function updateAllJammingZones(zonesData) {
  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "scene3d.js:updateAllJammingZones:entry",
      message: "Jamming zones update called",
      data: {
        zonesDataLength: zonesData?.length,
        isArray: Array.isArray(zonesData),
        existingZoneIds: Object.keys(jammingZones),
        zonesData: zonesData?.map((z) => ({ id: z.id, active: z.active })),
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      hypothesisId: "C",
    }),
  }).catch(() => {});
  // #endregion

  // Safety check - don't clear all zones if we get empty/invalid data
  // This prevents zones from disappearing on API errors
  if (!zonesData || !Array.isArray(zonesData)) {
    console.warn(
      "[Scene3D] Invalid jamming zones data, keeping existing zones",
    );
    return;
  }

  const zoneIds = new Set();

  for (const zone of zonesData) {
    if (zone && zone.id) {
      updateJammingZone(zone.id, zone);
      zoneIds.add(zone.id);
    }
  }

  // Only remove zones if we actually received valid data
  // Don't remove if zonesData was empty (could be an API error)
  if (zonesData.length > 0) {
    for (const zoneId of Object.keys(jammingZones)) {
      if (!zoneIds.has(zoneId)) {
        removeJammingZone(zoneId);
      }
    }
  }
}

// ============================================================================
// SPOOFING ZONE VISUALIZATION
// ============================================================================

const SPOOF_COLORS = {
  phantom: 0x9333ea,
  position_falsification: 0xec4899,
  coordinate: 0x7c3aed,
};

function updateSpoofingZone(zoneId, zoneData) {
  const center = zoneData.center || [0, 0, 0];
  const radius = zoneData.radius || 10;
  const spoofType = zoneData.spoof_type || "phantom";
  const color = SPOOF_COLORS[spoofType] || 0x9333ea;

  if (!spoofingZones[zoneId]) {
    const group = new THREE.Group();
    group.name = zoneId;

    // Outer wireframe sphere
    const outerGeo = new THREE.SphereGeometry(radius, 24, 16);
    const outerMat = new THREE.MeshBasicMaterial({
      color: color,
      wireframe: true,
      transparent: true,
      opacity: 0.2,
    });
    const outerSphere = new THREE.Mesh(outerGeo, outerMat);
    outerSphere.name = "outerSphere";
    group.add(outerSphere);

    // Inner solid sphere
    const innerGeo = new THREE.SphereGeometry(radius * 0.3, 16, 12);
    const innerMat = new THREE.MeshStandardMaterial({
      color: color,
      transparent: true,
      opacity: 0.35,
      emissive: color,
      emissiveIntensity: 0.3,
    });
    const innerSphere = new THREE.Mesh(innerGeo, innerMat);
    innerSphere.name = "innerSphere";
    group.add(innerSphere);

    // Label
    const typeLabels = {
      phantom: "PHANTOM",
      position_falsification: "POS FALSIFY",
      coordinate: "COORD ATTACK",
    };
    const labelText = `${typeLabels[spoofType] || spoofType} r=${radius}`;
    const label = createTextSprite(labelText, color, 10);
    label.position.set(0, radius + 2, 0);
    label.name = "label";
    group.add(label);

    scene.add(group);
    spoofingZones[zoneId] = group;
  }

  const zone = spoofingZones[zoneId];
  zone.position.set(-center[0], center[2] || 0, center[1]);

  // Slowly rotate wireframe for visual effect
  const outerSphere = zone.getObjectByName("outerSphere");
  if (outerSphere) {
    outerSphere.rotation.y += 0.005;
    outerSphere.rotation.x += 0.002;
  }
}

function removeSpoofingZone(zoneId) {
  if (spoofingZones[zoneId]) {
    scene.remove(spoofingZones[zoneId]);
    delete spoofingZones[zoneId];
  }
}

function updateAllSpoofingZones(zonesData) {
  if (!zonesData || !Array.isArray(zonesData)) return;

  const zoneIds = new Set();
  for (const zone of zonesData) {
    if (zone && zone.id) {
      updateSpoofingZone(zone.id, zone);
      zoneIds.add(zone.id);
    }
  }

  for (const zoneId of Object.keys(spoofingZones)) {
    if (!zoneIds.has(zoneId)) {
      removeSpoofingZone(zoneId);
    }
  }
}

// ============================================================================
// PHANTOM AGENT & FALSIFICATION VISUALIZATION
// ============================================================================

function updatePhantomAgents(phantomAgents) {
  if (!phantomAgents) {
    // Clear all phantoms
    for (const pid of Object.keys(phantomVehicles)) {
      scene.remove(phantomVehicles[pid]);
    }
    phantomVehicles = {};
    return;
  }

  const phantomIds = new Set(Object.keys(phantomAgents));

  for (const [phantomId, pos] of Object.entries(phantomAgents)) {
    if (!phantomVehicles[phantomId]) {
      // Create translucent phantom vehicle
      const group = new THREE.Group();
      group.name = phantomId;

      const bodyGeo = new THREE.BoxGeometry(0.8, 0.4, 1.2);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x9333ea,
        transparent: true,
        opacity: 0.35,
        emissive: 0x9333ea,
        emissiveIntensity: 0.2,
      });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.position.y = 0.3;
      body.name = "body";
      group.add(body);

      // Ghost glow ring
      const ringGeo = new THREE.TorusGeometry(0.8, 0.04, 8, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x9333ea,
        transparent: true,
        opacity: 0.4,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = 0.1;
      ring.name = "ghostRing";
      group.add(ring);

      // Label
      const label = createTextSprite(phantomId.replace(/_/g, " "), 0x9333ea, 9);
      label.position.set(0, 1.8, 0);
      group.add(label);

      scene.add(group);
      phantomVehicles[phantomId] = group;
    }

    // Update position
    const vehicle = phantomVehicles[phantomId];
    vehicle.position.set(-pos[0], pos[2] || 0, pos[1]);

    // Animate ghost ring pulsing
    const ring = vehicle.getObjectByName("ghostRing");
    if (ring) {
      ring.material.opacity = 0.25 + Math.sin(Date.now() * 0.003) * 0.15;
    }
    // Flicker body opacity
    const body = vehicle.getObjectByName("body");
    if (body) {
      body.material.opacity = 0.25 + Math.sin(Date.now() * 0.005 + 1) * 0.1;
    }
  }

  // Remove phantoms no longer present
  for (const pid of Object.keys(phantomVehicles)) {
    if (!phantomIds.has(pid)) {
      scene.remove(phantomVehicles[pid]);
      delete phantomVehicles[pid];
    }
  }
}

function updateFalsificationLines(offsets, agentPositions) {
  // Clear old lines
  for (const lineObj of Object.values(falsificationLines)) {
    scene.remove(lineObj);
  }
  falsificationLines = {};

  if (!offsets || !agentPositions) return;

  for (const [agentId, offset] of Object.entries(offsets)) {
    const truePos = agentPositions[agentId];
    if (!truePos) continue;

    const spoofedPos = [
      truePos[0] + offset[0],
      truePos[1] + offset[1],
      truePos[2] + offset[2],
    ];

    // Dashed red line from true to spoofed position
    const material = new THREE.LineDashedMaterial({
      color: 0xff4444,
      dashSize: 0.5,
      gapSize: 0.3,
      transparent: true,
      opacity: 0.7,
    });

    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-truePos[0], truePos[2] || 0, truePos[1]),
      new THREE.Vector3(-spoofedPos[0], spoofedPos[2] || 0, spoofedPos[1]),
    ]);

    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    scene.add(line);
    falsificationLines[agentId] = line;
  }
}

function updateCryptoShields(enabled) {
  for (const [agentId, vehicle] of Object.entries(vehicles)) {
    let shield = vehicle.getObjectByName("cryptoShield");
    if (enabled) {
      if (!shield) {
        const shieldGeo = new THREE.TorusGeometry(0.6, 0.03, 6, 20);
        const shieldMat = new THREE.MeshBasicMaterial({
          color: 0x22c55e,
          transparent: true,
          opacity: 0.5,
        });
        shield = new THREE.Mesh(shieldGeo, shieldMat);
        shield.rotation.x = Math.PI / 2;
        shield.position.y = 1.5;
        shield.name = "cryptoShield";
        vehicle.add(shield);
      }
      shield.visible = true;
      shield.material.opacity = 0.3 + Math.sin(Date.now() * 0.002) * 0.15;
    } else if (shield) {
      shield.visible = false;
    }
  }
}

// ============================================================================
// COMMUNICATION LINKS VISUALIZATION
// ============================================================================

/**
 * Update communication links between agents
 * Shows dotted lines for agents with aij >= PT (neighbors)
 */
function updateCommunicationLinks(links, agentPositions) {
  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "scene3d.js:updateCommunicationLinks:entry",
      message: "Communication links update called",
      data: {
        linksCount: links?.length,
        hasAgentPositions: !!agentPositions,
        agentPositionsKeys: Object.keys(agentPositions || {}),
        linksData: links?.slice(0, 3),
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      hypothesisId: "A,B",
    }),
  }).catch(() => {});
  // #endregion

  // Safely remove old links - defer disposal to avoid render conflicts
  const oldLinks = [...communicationLinks];
  communicationLinks = [];

  // Remove from scene immediately
  for (const link of oldLinks) {
    scene.remove(link);
  }

  // Skip disposal - let GC handle it to avoid render conflicts
  // Materials and geometries will be cleaned up when no longer referenced

  if (!links || links.length === 0) return;

  for (const link of links) {
    const fromPos = agentPositions[link.from];
    const toPos = agentPositions[link.to];

    if (!fromPos || !toPos) continue;

    // Determine color based on link strength and quality
    const quality = link.quality || 0;
    const isStrong = link.strong !== false; // Default to true for backwards compatibility

    let color;
    let dashSize = 1;
    let gapSize = 0.5;
    let opacity;

    if (isStrong) {
      // Strong link (aij >= PT) - green
      color =
        quality > 0.7 ? CONFIG.colors.commLink : CONFIG.colors.commLinkWeak;
      opacity = 0.5 + quality * 0.3;
    } else {
      // Weak/degraded link (aij < PT) - red/orange, more dashed
      color = 0xff6b6b; // Red-ish for degraded
      dashSize = 0.5;
      gapSize = 1.0;
      opacity = 0.4 + quality * 0.3;
    }

    // Create dashed line material with MAX visibility
    const material = new THREE.LineDashedMaterial({
      color: color,
      dashSize: dashSize,
      gapSize: gapSize,
      transparent: true,
      opacity: opacity,
      depthTest: true,
      depthWrite: true,
    });

    // Create line geometry
    // Convert sim coords to Three.js: [-x, z, y] (negate X)
    const points = [
      new THREE.Vector3(-fromPos[0], fromPos[2] || 0, fromPos[1]),
      new THREE.Vector3(-toPos[0], toPos[2] || 0, toPos[1]),
    ];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    line.renderOrder = 50; // Render above most objects

    scene.add(line);
    communicationLinks.push(line);

    // #region agent log
    if (communicationLinks.length === 1) {
      // Log first link creation
      fetch(
        "http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            location: "scene3d.js:updateCommunicationLinks:created",
            message: "Communication link CREATED",
            data: {
              from: link.from,
              to: link.to,
              color: color,
              opacity: opacity,
              inScene: scene.children.includes(line),
            },
            timestamp: Date.now(),
            sessionId: "debug-session",
            hypothesisId: "A-render",
          }),
        },
      ).catch(() => {});
    }
    // #endregion
  }
}

// ============================================================================
// WAYPOINT PATH VISUALIZATION (disabled - waypoints shown only in minimap)
// ============================================================================

/**
 * Update waypoint paths - DISABLED for 3D scene
 * Waypoints are now only shown in the minimap, not in 3D
 */
function updateWaypointPaths(waypointsData) {
  // Clear any existing waypoint paths from 3D scene
  for (const agentId of Object.keys(waypointPaths)) {
    const pathGroup = waypointPaths[agentId];
    if (pathGroup && scene) {
      try {
        scene.remove(pathGroup);
      } catch (e) {
        // Ignore errors
      }
    }
  }
  waypointPaths = {};
  // Waypoints are now only displayed in the minimap (see app.js MiniMap)
}

/**
 * Update traveled path trails for all agents
 * Shows the path each agent has traveled
 */
function updateTraveledPaths(pathsData) {
  // Safely remove old trail lines
  const oldPaths = { ...traveledPaths };
  traveledPaths = {};

  // Remove from scene immediately with null safety
  for (const agentId of Object.keys(oldPaths)) {
    const pathLine = oldPaths[agentId];
    if (pathLine && scene) {
      try {
        scene.remove(pathLine);
      } catch (e) {
        console.warn(`[Scene3D] Error removing trail for ${agentId}:`, e);
      }
    }
  }

  // Null safety - if no data or trails disabled, just clear and return
  if (!pathsData || !showTrails || Object.keys(pathsData).length === 0) return;

  // Colors for different agents (cycling through)
  const trailColors = [
    0xfbbf24, // amber
    0x34d399, // emerald
    0x60a5fa, // blue
    0xf472b6, // pink
    0xa78bfa, // violet
    0xfb923c, // orange
  ];

  let colorIndex = 0;
  for (const [agentId, path] of Object.entries(pathsData)) {
    if (!path || path.length < 2) continue;

    // Create trail line from path history (negate X)
    const points = path.map(
      (p) => new THREE.Vector3(-p[0], (p[2] || 0) + 0.3, p[1]), // Slightly above ground
    );

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: trailColors[colorIndex % trailColors.length],
      transparent: true,
      opacity: 0.6,
    });

    const trailLine = new THREE.Line(geometry, material);
    trailLine.name = `trail_${agentId}`;

    scene.add(trailLine);
    traveledPaths[agentId] = trailLine;
    colorIndex++;
  }
}

/**
 * Set waypoints visibility - NO-OP (waypoints now only in minimap)
 */
function setWaypointsVisible(visible) {
  // Waypoints are now only shown in minimap, not 3D scene
  showWaypoints = visible;
}

/**
 * Set trails visibility
 */
function setTrailsVisible(visible) {
  showTrails = visible;
  for (const path of Object.values(traveledPaths)) {
    if (path) path.visible = visible;
  }
}

/**
 * Update LLM guidance arrows on vehicles
 * Shows purple/cyan arrows indicating where LLM is directing agents
 */
function updateLLMGuidanceArrows(guidanceData, agentPositions) {
  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "scene3d.js:updateLLMGuidanceArrows:entry",
      message: "LLM guidance arrows update called",
      data: {
        guidanceCount: guidanceData?.length,
        hasAgentPositions: !!agentPositions,
        guidanceData: guidanceData,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      hypothesisId: "D,F",
    }),
  }).catch(() => {});
  // #endregion

  // Safely remove old arrows
  const oldArrows = { ...llmGuidanceArrows };
  llmGuidanceArrows = {};

  // Remove from scene immediately
  for (const agentId of Object.keys(oldArrows)) {
    const arrow = oldArrows[agentId];
    if (arrow) {
      scene.remove(arrow);
    }
  }

  // Skip disposal - let GC handle it to avoid render conflicts
  // Materials and geometries will be cleaned up when no longer referenced

  if (!guidanceData || guidanceData.length === 0) return;

  for (const guidance of guidanceData) {
    const { agent_id, direction, speed } = guidance;
    const pos = agentPositions[agent_id];

    if (!pos || !direction) {
      console.warn(
        `[Scene3D] LLM guidance missing pos or direction for ${agent_id}`,
        { pos, direction },
      );
      continue;
    }

    // Create arrow from vehicle position pointing in direction
    // Position arrow above the vehicle for visibility (negate X)
    const origin = new THREE.Vector3(-pos[0], (pos[2] || 0) + 3, pos[1]);

    // Convert direction (x, y, z) to Three.js coordinates (-x, z, y)
    const dir = new THREE.Vector3(
      -direction[0],
      direction[2] || 0,
      direction[1],
    );

    // Ensure direction is valid and normalized
    if (dir.length() < 0.001) {
      dir.set(0, 1, 0); // Default up if zero
    } else {
      dir.normalize();
    }

    // Arrow length based on speed - make it more visible
    const length = 5 + speed * 6;

    // Create arrow helper with more visible settings
    const arrowColor = CONFIG.colors.llmGuidance;
    const arrowHelper = new THREE.ArrowHelper(
      dir,
      origin,
      length,
      arrowColor,
      length * 0.35, // Head length - larger
      length * 0.25, // Head width - larger
    );

    // Make it visible (don't set properties that don't exist on BasicMaterial)
    if (arrowHelper.line && arrowHelper.line.material) {
      arrowHelper.line.material.transparent = true;
      arrowHelper.line.material.opacity = 1.0;
    }
    if (arrowHelper.cone && arrowHelper.cone.material) {
      arrowHelper.cone.material.transparent = true;
      arrowHelper.cone.material.opacity = 1.0;
    }

    scene.add(arrowHelper);
    llmGuidanceArrows[agent_id] = arrowHelper;

    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "scene3d.js:updateLLMGuidanceArrows:created",
        message: "LLM guidance arrow CREATED",
        data: {
          agent_id: agent_id,
          origin: origin.toArray(),
          dir: dir.toArray(),
          length: length,
        },
        timestamp: Date.now(),
        sessionId: "debug-session",
        hypothesisId: "D-render",
      }),
    }).catch(() => {});
    // #endregion
  }
}

/**
 * Update trajectory lines for chat-commanded LLM targets
 * Shows dashed line from agent to target coordinate
 */
function updateLLMTargetLines(llmTargets, agentPositions) {
  // Safely remove old lines
  const oldLines = { ...llmTargetLines };
  llmTargetLines = {};

  // Remove from scene immediately
  for (const agentId of Object.keys(oldLines)) {
    const lineObj = oldLines[agentId];
    if (lineObj) {
      scene.remove(lineObj);
    }
  }

  if (!llmTargets || Object.keys(llmTargets).length === 0) return;
  if (!agentPositions) return;

  for (const [agentId, target] of Object.entries(llmTargets)) {
    const pos = agentPositions[agentId];
    if (!pos || !target) continue;

    // Create dashed line from agent to target (negate X for Three.js coords)
    const startPos = new THREE.Vector3(-pos[0], (pos[2] || 0) + 1, pos[1]);
    const endPos = new THREE.Vector3(
      -target[0],
      (target[2] || 0) + 1,
      target[1],
    );

    const points = [startPos, endPos];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    // Dashed line material (cyan/purple color)
    const material = new THREE.LineDashedMaterial({
      color: 0x4fc3f7, // Light cyan
      dashSize: 2,
      gapSize: 1,
      linewidth: 2,
      transparent: true,
      opacity: 0.8,
    });

    const line = new THREE.Line(geometry, material);
    line.computeLineDistances(); // Required for dashed lines
    line.name = `llm_target_${agentId}`;

    // Add a small sphere at the target position
    const targetMarkerGeom = new THREE.SphereGeometry(0.8, 8, 8);
    const targetMarkerMat = new THREE.MeshBasicMaterial({
      color: 0x4fc3f7,
      transparent: true,
      opacity: 0.8,
    });
    const targetMarker = new THREE.Mesh(targetMarkerGeom, targetMarkerMat);
    targetMarker.position.copy(endPos);

    // Create group to hold line and marker
    const group = new THREE.Group();
    group.add(line);
    group.add(targetMarker);
    group.name = `llm_target_group_${agentId}`;

    scene.add(group);
    llmTargetLines[agentId] = group;

    console.log(
      `[Scene3D] Created trajectory line for ${agentId} to (${target[0]}, ${target[1]}, ${target[2]})`,
    );
  }
}

/**
 * Update control vectors showing NET control direction when avoidance is active
 * Shows the actual direction the vehicle will move (aligns with heading)
 * Only displayed when obstacle avoidance is influencing movement
 * @param {Object} avoidanceData - Control vector data per agent (named for API compatibility)
 * @param {Object} agentPositions - Position data per agent
 * @param {Set} llmAssistedAgents - Set of agent IDs with active LLM guidance (to skip)
 */
function updateAvoidanceVectors(
  avoidanceData,
  agentPositions,
  llmAssistedAgents = new Set(),
) {
  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "scene3d.js:updateAvoidanceVectors:entry",
      message: "Avoidance vectors update called",
      data: {
        avoidanceDataKeys: Object.keys(avoidanceData || {}),
        hasAgentPositions: !!agentPositions,
        llmAssistedCount: llmAssistedAgents.size,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      hypothesisId: "E,F",
    }),
  }).catch(() => {});
  // #endregion

  // Safely remove old vectors
  const oldVectors = { ...avoidanceVectors };
  avoidanceVectors = {};

  // Remove from scene immediately
  for (const agentId of Object.keys(oldVectors)) {
    const arrow = oldVectors[agentId];
    if (arrow) {
      scene.remove(arrow);
    }
  }

  // Skip disposal - let GC handle it to avoid render conflicts
  // Materials and geometries will be cleaned up when no longer referenced

  if (!avoidanceData || Object.keys(avoidanceData).length === 0) return;

  for (const [agentId, avoidance] of Object.entries(avoidanceData)) {
    // Skip agents with active LLM guidance - LLM arrow takes precedence
    if (llmAssistedAgents.has(agentId)) {
      continue;
    }

    const pos = agentPositions[agentId];
    if (!pos || !avoidance || !avoidance.direction) continue;

    const { direction, magnitude } = avoidance;

    // Show avoidance vectors with lower threshold for better visibility
    if (magnitude < 0.01) continue;

    const origin = new THREE.Vector3(-pos[0], (pos[2] || 0) + 2, pos[1]);
    const dir = new THREE.Vector3(
      -direction[0],
      direction[2] || 0,
      direction[1],
    );

    // Ensure direction is valid
    if (dir.length() < 0.001) {
      dir.set(0, 1, 0);
    } else {
      dir.normalize();
    }

    // Scale length reasonably based on magnitude
    const length = Math.max(3, 2 + magnitude * 3);

    const arrowHelper = new THREE.ArrowHelper(
      dir,
      origin,
      length,
      CONFIG.colors.avoidance,
      length * 0.3,
      length * 0.2,
    );

    // Make arrow visible (only set properties that exist on BasicMaterial)
    if (arrowHelper.line && arrowHelper.line.material) {
      arrowHelper.line.material.transparent = true;
      arrowHelper.line.material.opacity = 1.0;
    }
    if (arrowHelper.cone && arrowHelper.cone.material) {
      arrowHelper.cone.material.transparent = true;
      arrowHelper.cone.material.opacity = 1.0;
    }

    scene.add(arrowHelper);
    avoidanceVectors[agentId] = arrowHelper;

    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "scene3d.js:updateAvoidanceVectors:created",
        message: "Avoidance vector CREATED",
        data: {
          agent_id: agentId,
          origin: origin.toArray(),
          dir: dir.toArray(),
          length: length,
          magnitude: magnitude,
        },
        timestamp: Date.now(),
        sessionId: "debug-session",
        hypothesisId: "E-render",
      }),
    }).catch(() => {});
    // #endregion
  }
}

/**
 * Update all visualization (links + waypoints + LLM guidance + avoidance)
 */
function updateVisualization(visData) {
  if (!visData) {
    console.warn("[Scene3D] updateVisualization called with no data");
    return;
  }

  // Update communication links (only if we have valid data)
  if (
    visData.communication_links &&
    visData.agent_positions &&
    Object.keys(visData.agent_positions).length > 0
  ) {
    updateCommunicationLinks(
      visData.communication_links,
      visData.agent_positions,
    );
  }
  // If agent_positions is empty, keep existing visualization (timing issue)

  // Waypoints are now only shown in minimap (not 3D scene)
  // Clear any stale waypoint paths from 3D
  updateWaypointPaths(null);

  // Update traveled path trails
  if (visData.traveled_paths) {
    updateTraveledPaths(visData.traveled_paths);
  }

  // Check if we have valid agent positions (guard against timing issues)
  const hasValidPositions =
    visData.agent_positions && Object.keys(visData.agent_positions).length > 0;

  // Update LLM guidance arrows
  // Track which agents have LLM guidance so avoidance vectors can skip them
  const llmAssistedAgents = new Set();
  if (visData.llm_guidance && hasValidPositions) {
    if (visData.llm_guidance.length > 0) {
      console.log(
        "[Scene3D] Updating LLM guidance arrows:",
        visData.llm_guidance.length,
        "entries",
      );
      // Collect agent IDs with active LLM guidance
      visData.llm_guidance.forEach((g) => llmAssistedAgents.add(g.agent_id));
    }
    updateLLMGuidanceArrows(visData.llm_guidance, visData.agent_positions);
  }

  // Update control vectors (skip agents with active LLM guidance)
  // These show NET control direction when avoidance is active
  if (visData.avoidance_vectors && hasValidPositions) {
    const controlCount = Object.keys(visData.avoidance_vectors).length;
    if (controlCount > 0) {
      console.log(
        "[Scene3D] Updating control vectors:",
        controlCount,
        "agents",
      );
    }
    updateAvoidanceVectors(
      visData.avoidance_vectors,
      visData.agent_positions,
      llmAssistedAgents,
    );
  }

  // Update chat-commanded trajectory lines (llm_targets)
  if (visData.llm_targets && hasValidPositions) {
    updateLLMTargetLines(visData.llm_targets, visData.agent_positions);
  }

  // Update phantom agents from MAVLink spoofing
  if (visData.phantom_agents) {
    updatePhantomAgents(visData.phantom_agents);
  }

  // Update falsification offset lines
  if (visData.falsification_offsets && hasValidPositions) {
    updateFalsificationLines(visData.falsification_offsets, visData.agent_positions);
  }

  // Update crypto auth shield indicators on agents
  if (visData.crypto_auth_enabled !== undefined) {
    updateCryptoShields(visData.crypto_auth_enabled);
  }
}

/**
 * Mouse move handler for hover effects
 */
function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // Check for vehicle intersections
  const vehicleObjects = Object.values(vehicles).flatMap((v) =>
    v.children.filter((c) => c.name === "body"),
  );
  const vehicleIntersects = raycaster.intersectObjects(vehicleObjects);

  if (vehicleIntersects.length > 0) {
    const hitVehicle = vehicleIntersects[0].object.parent;
    const agentId = hitVehicle.name;

    if (hoveredVehicle !== agentId) {
      hoveredVehicle = agentId;
      showTooltip(event, hitVehicle.userData.data);
    } else {
      updateTooltipPosition(event);
    }

    renderer.domElement.style.cursor = "pointer";
    return;
  }

  // Check for destination marker intersection
  if (missionMarker) {
    const destIntersects = raycaster.intersectObject(missionMarker);
    if (destIntersects.length > 0) {
      showDestinationTooltip(event);
      renderer.domElement.style.cursor = "pointer";
      return;
    }
  }

  // Nothing hovered - but keep tooltip if vehicle is selected
  hoveredVehicle = null;
  if (!selectedVehicle) {
    hideTooltip();
  }
  renderer.domElement.style.cursor = "default";
}

/**
 * Show tooltip for destination marker
 */
function showDestinationTooltip(event) {
  if (!tooltip) return;

  const [mx, my, mz] = CONFIG.missionEnd;

  tooltip.innerHTML = `
    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
      <span style="font-weight: 600; font-size: 13px; color: #38bdf8;">Destination</span>
    </div>
    <div style="display: grid; grid-template-columns: auto auto; gap: 4px 16px; font-size: 11px;">
      <span style="color: hsl(240 5% 64.9%);">X</span><span>${mx}</span>
      <span style="color: hsl(240 5% 64.9%);">Y</span><span>${my}</span>
      <span style="color: hsl(240 5% 64.9%);">Z</span><span>${mz}</span>
    </div>
  `;

  tooltip.style.display = "block";
  tooltip.style.left = event.clientX + 15 + "px";
  tooltip.style.top = event.clientY + 15 + "px";
}

/**
 * Mouse click handler for selection
 */
function onMouseClick(event) {
  if (hoveredVehicle) {
    // Select/deselect vehicle
    if (selectedVehicle === hoveredVehicle) {
      selectedVehicle = null;
      hideTooltip();
    } else {
      selectedVehicle = hoveredVehicle;
      // Keep tooltip visible for selected vehicle
      const vehicle = vehicles[selectedVehicle];
      if (vehicle && vehicle.userData.data) {
        showTooltip(event, vehicle.userData.data);
      }
    }

    // Update selection rings
    for (const [agentId, vehicle] of Object.entries(vehicles)) {
      const ring = vehicle.getObjectByName("selectRing");
      if (ring) ring.visible = agentId === selectedVehicle;
    }

    // Dispatch custom event
    window.dispatchEvent(
      new CustomEvent("vehicleSelected", {
        detail: { agentId: selectedVehicle },
      }),
    );
  } else {
    // Clicked on empty space - deselect
    if (selectedVehicle) {
      selectedVehicle = null;
      hideTooltip();
      for (const vehicle of Object.values(vehicles)) {
        const ring = vehicle.getObjectByName("selectRing");
        if (ring) ring.visible = false;
      }
    }
  }
}

/**
 * Show tooltip with vehicle data
 */
function showTooltip(event, data) {
  if (!tooltip || !data) return;

  const pos = data.position || [0, 0, 0];
  const vel = data.velocity || [0, 0, 0];
  const speed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2);
  const statusColor = data.jammed ? "#ef4444" : "#22c55e";
  const statusText = data.jammed ? "JAMMED" : "OK";

  tooltip.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
      <span style="font-weight: 600; font-size: 13px;">${formatDisplayName(data.agent_id) || "Unknown"}</span>
      <span style="font-size: 10px; padding: 2px 6px; border-radius: 4px; background: ${statusColor}20; color: ${statusColor};">${statusText}</span>
    </div>
    <div style="display: grid; grid-template-columns: auto auto; gap: 4px 16px; font-size: 11px;">
      <span style="color: hsl(240 5% 64.9%);">Position</span>
      <span>(${pos[0]?.toFixed(1)}, ${pos[1]?.toFixed(1)}, ${pos[2]?.toFixed(1)})</span>
      <span style="color: hsl(240 5% 64.9%);">Speed</span>
      <span>${speed.toFixed(2)} m/s</span>
      <span style="color: hsl(240 5% 64.9%);">Heading</span>
      <span>${(data.heading_degrees || 0).toFixed(0)}°</span>
      <span style="color: hsl(240 5% 64.9%);">Comm</span>
      <span>${((data.communication_quality || 0) * 100).toFixed(0)}%</span>
      ${data.distance_to_goal ? `<span style="color: hsl(240 5% 64.9%);">To Goal</span><span>${data.distance_to_goal.toFixed(1)} m</span>` : ""}
    </div>
    ${
      data.llm_target
        ? `
      <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid hsl(240 3.7% 20%); font-size: 10px; color: #facc15;">
        Target: (${data.llm_target[0]?.toFixed(1)}, ${data.llm_target[1]?.toFixed(1)}, ${(data.llm_target[2] || 0).toFixed(1)})
      </div>
    `
        : ""
    }
  `;

  tooltip.style.display = "block";
  updateTooltipPosition(event);
}

/**
 * Update tooltip position
 */
function updateTooltipPosition(event) {
  if (!tooltip) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const x = event.clientX - rect.left + 15;
  const y = event.clientY - rect.top + 15;

  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
}

/**
 * Hide tooltip
 */
function hideTooltip() {
  if (tooltip) tooltip.style.display = "none";
}

/**
 * Animation loop
 */
function animate() {
  requestAnimationFrame(animate);

  controls.update();

  // Rotate mission marker
  if (missionMarker) {
    missionMarker.rotation.y += 0.01;
  }

  // Animate status lights
  const time = Date.now() * 0.003;
  for (const vehicle of Object.values(vehicles)) {
    const light = vehicle.getObjectByName("statusLight");
    if (light) {
      light.material.emissiveIntensity = 0.3 + Math.sin(time) * 0.2;
    }

    // Animate selection ring
    const ring = vehicle.getObjectByName("selectRing");
    if (ring && ring.visible) {
      ring.rotation.z += 0.02;
    }
  }

  // Animate jamming zones
  for (const zone of Object.values(jammingZones)) {
    const wireframe = zone.getObjectByName("wireframe");
    if (wireframe) {
      wireframe.rotation.y += 0.005;
      wireframe.rotation.x = Math.sin(time * 0.5) * 0.1;
    }
  }

  // Render main scene with error protection
  try {
    renderer.render(scene, camera);
  } catch (e) {
    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/eea87f04-42bd-46c8-90dc-83ca820d957d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: "scene3d.js:render:error",
        message: "RENDER ERROR CAUGHT",
        data: {
          error: e.message,
          stack: e.stack?.substring(0, 500),
          sceneChildCount: scene.children.length,
        },
        timestamp: Date.now(),
        sessionId: "debug-session",
        hypothesisId: "H1-H4",
      }),
    }).catch(() => {});
    // #endregion
    console.error("[Scene3D] Render error:", e);
    // Find and remove problematic objects
    const toRemove = [];
    scene.traverse((obj) => {
      if (obj.material && !obj.material.uuid) {
        toRemove.push(obj);
      }
    });
    toRemove.forEach((obj) => scene.remove(obj));
  }

  // Render axis indicator synced with main camera rotation
  if (axisRenderer && axisScene && axisCamera) {
    // Copy camera rotation to axis scene
    axisCamera.position.copy(camera.position);
    axisCamera.position.sub(controls.target);
    axisCamera.position.normalize().multiplyScalar(4);
    axisCamera.lookAt(0, 0, 0);

    try {
      axisRenderer.render(axisScene, axisCamera);
    } catch (e) {
      console.error("[Scene3D] Axis render error:", e);
    }
  }
}

/**
 * Get selected vehicle
 */
function getSelectedVehicle() {
  return selectedVehicle;
}

/**
 * Select a vehicle programmatically (from sidebar)
 */
function selectVehicle(agentId) {
  // Deselect previous
  if (selectedVehicle && vehicles[selectedVehicle]) {
    const prevRing = vehicles[selectedVehicle].getObjectByName("selectRing");
    if (prevRing) prevRing.visible = false;
  }

  // Select new
  selectedVehicle = agentId;

  if (agentId && vehicles[agentId]) {
    const vehicle = vehicles[agentId];
    const ring = vehicle.getObjectByName("selectRing");
    if (ring) ring.visible = true;

    // Center camera on vehicle
    if (controls && vehicle.position) {
      controls.target.copy(vehicle.position);
    }

    // Show tooltip
    if (vehicle.userData.data) {
      // Position tooltip near center of screen
      const fakeEvent = {
        clientX: window.innerWidth / 2,
        clientY: window.innerHeight / 3,
      };
      showTooltip(fakeEvent, vehicle.userData.data);
    }

    // Dispatch event
    window.dispatchEvent(
      new CustomEvent("vehicleSelected", { detail: { agentId } }),
    );
  } else {
    hideTooltip();
  }
}

/**
 * Get selected jamming zone
 */
function getSelectedJammingZone() {
  return selectedJammingZone;
}

/**
 * Select a jamming zone programmatically (from sidebar)
 */
function selectJammingZone(zoneId) {
  // Deselect previous
  if (selectedJammingZone && jammingZones[selectedJammingZone]) {
    const prevZone = jammingZones[selectedJammingZone];
    const sphere = prevZone.getObjectByName("sphere");
    const wireframe = prevZone.getObjectByName("wireframe");
    if (sphere) {
      sphere.material.opacity = 0.15;
      sphere.material.emissive?.setHex(0x000000);
    }
    if (wireframe) {
      wireframe.material.opacity = 0.3;
    }
  }

  // Select new
  selectedJammingZone = zoneId;

  if (zoneId && jammingZones[zoneId]) {
    const zone = jammingZones[zoneId];
    const sphere = zone.getObjectByName("sphere");
    const wireframe = zone.getObjectByName("wireframe");

    // Highlight selected zone
    if (sphere) {
      sphere.material.opacity = 0.35;
      if (sphere.material.emissive) {
        sphere.material.emissive.setHex(0xff4444);
      }
    }
    if (wireframe) {
      wireframe.material.opacity = 0.6;
    }

    // Center camera on zone
    if (controls && zone.position) {
      controls.target.copy(zone.position);
    }

    // Show tooltip for zone
    const fakeEvent = {
      clientX: window.innerWidth / 2,
      clientY: window.innerHeight / 3,
    };
    showJammingZoneTooltip(fakeEvent, zoneId, zone.position);

    // Dispatch event
    window.dispatchEvent(
      new CustomEvent("jammingZoneSelected", { detail: { zoneId } }),
    );
  } else {
    hideTooltip();
  }
}

/**
 * Select a spoofing zone programmatically (from sidebar)
 */
function selectSpoofingZone(zoneId) {
  // Deselect previous
  if (selectedSpoofingZone && spoofingZones[selectedSpoofingZone]) {
    const prevZone = spoofingZones[selectedSpoofingZone];
    const outerSphere = prevZone.getObjectByName("outerSphere");
    if (outerSphere) {
      outerSphere.material.opacity = 0.15;
    }
  }

  selectedSpoofingZone = zoneId;

  if (zoneId && spoofingZones[zoneId]) {
    const zone = spoofingZones[zoneId];
    const outerSphere = zone.getObjectByName("outerSphere");

    // Highlight selected zone
    if (outerSphere) {
      outerSphere.material.opacity = 0.5;
    }

    // Center camera on zone
    if (controls && zone.position) {
      controls.target.copy(zone.position);
    }

    // Show a simple tooltip
    if (tooltip) {
      tooltip.innerHTML = `
        <div style="font-weight: 600; color: #c084fc; margin-bottom: 6px;">
          ${formatDisplayName(zoneId)}
        </div>
        <div style="font-size: 11px; color: hsl(240 5% 64.9%);">
          Position: (${zone.position.x.toFixed(0)}, ${zone.position.z.toFixed(0)}, ${zone.position.y.toFixed(0)})
        </div>
      `;
      tooltip.style.display = "block";
      tooltip.style.left = `${window.innerWidth / 2 + 15}px`;
      tooltip.style.top = `${window.innerHeight / 3 + 15}px`;
    }

    window.dispatchEvent(
      new CustomEvent("spoofingZoneSelected", { detail: { zoneId } })
    );
  } else {
    hideTooltip();
  }
}

/**
 * Show tooltip for jamming zone
 */
function showJammingZoneTooltip(event, zoneId, position) {
  if (!tooltip) return;

  tooltip.innerHTML = `
    <div style="font-weight: 600; color: #f87171; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
      ${formatDisplayName(zoneId)}
    </div>
    <div style="display: grid; grid-template-columns: auto auto; gap: 4px 12px; font-size: 11px;">
      <span style="color: hsl(240 5% 64.9%);">Position</span>
      <span>(${position.x.toFixed(0)}, ${position.z.toFixed(0)}, ${position.y.toFixed(0)})</span>
    </div>
  `;

  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 15}px`;
  tooltip.style.top = `${event.clientY + 15}px`;
}

/**
 * Get current configuration (for mini-map etc.)
 */
function getConfig() {
  return {
    bounds: CONFIG.bounds,
    missionEnd: CONFIG.missionEnd,
    colors: CONFIG.colors,
  };
}

/**
 * Get all vehicle positions and headings
 */
function getAllVehicleStates() {
  const states = [];
  for (const [agentId, vehicle] of Object.entries(vehicles)) {
    if (vehicle && vehicle.position) {
      states.push({
        id: agentId,
        position: [-vehicle.position.x, vehicle.position.z, vehicle.position.y], // Convert back to sim coords (negate X)
        heading: vehicle.rotation ? -vehicle.rotation.y + Math.PI / 2 : 0,
      });
    }
  }
  return states;
}

/**
 * Get all jamming zone data
 */
function getAllJammingZones() {
  const zones = [];
  for (const [zoneId, zone] of Object.entries(jammingZones)) {
    if (zone && zone.position) {
      // Get radius from sphere child
      const sphere = zone.getObjectByName("sphere");
      const radius = sphere?.geometry?.parameters?.radius || 10;
      zones.push({
        id: zoneId,
        center: [-zone.position.x, zone.position.z, zone.position.y], // Convert back to sim coords (negate X)
        radius: radius,
        active: zone.visible,
      });
    }
  }
  return zones;
}

// Export functions
window.Scene3D = {
  init: initScene,
  updateVehicle,
  updateAllVehicles,
  removeVehicle,
  updateJammingZone,
  updateAllJammingZones,
  removeJammingZone,
  updateSpoofingZone,
  updateAllSpoofingZones,
  removeSpoofingZone,
  updatePhantomAgents,
  updateFalsificationLines,
  updateCryptoShields,
  getSelectedVehicle,
  selectVehicle,
  getSelectedJammingZone,
  selectJammingZone,
  selectSpoofingZone,
  updateCommunicationLinks,
  updateWaypointPaths,
  updateTraveledPaths,
  updateLLMGuidanceArrows,
  updateLLMTargetLines,
  updateAvoidanceVectors,
  updateVisualization,
  setWaypointsVisible,
  setTrailsVisible,
  getConfig,
  getAllVehicleStates,
  getAllJammingZones,
};
