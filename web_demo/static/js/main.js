/**
 * Main application logic
 * Handles UI interactions, API calls, and 3D rendering loop
 */

class MotionApp {
    constructor() {
        this.isRunning = false;
        this.targetFps = 20; // Model generates data at 20fps
        this.frameInterval = 1000 / this.targetFps; // 50ms
        this.lastFetchTime = 0;
        this.nextFetchTime = 0;  // Scheduled time for next fetch
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.fpsUpdateTime = 0;
        this.lastRenderTime = 0;

        // Motion FPS tracking (frame consumption rate)
        this.motionFrameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdateTime = 0;

        // Request throttling
        this.isFetchingFrame = false;  // Prevent concurrent requests
        this.consecutiveWaiting = 0;   // Count consecutive 'waiting' responses

        // Session management
        this.sessionId = this.generateSessionId();

        // Camera follow settings
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000; // Auto-follow after 2 seconds of inactivity (reduced from 3s)
        this.currentRootPos = new THREE.Vector3(0, 1, 0);

        // Trajectory drawing (mouse drag on ground)
        this.initialTaskY = 1.0; // Used as y for xz-only trajectory points.
        this.taskYCaptured = false; // Capture initial root y on first frame after start.
        this.isDrawingTrajectory = false;
        this.drawnWaypoints = []; // Array of [x, y, z]
        this.drawnWaypointMinDist = 0.05; // Avoid duplicate points while dragging.
        this.trajPointSpheres = [];
        this.trajectoryPushThrottleMs = 120;
        this.lastTrajectoryPushTime = 0;
        this.trajectoryPushInFlight = false;
        this.pendingTrajectoryPush = false;

        this.initThreeJS();
        this.initUI();
        this.updateStatus();
        this.setupBeforeUnload();

        console.log('Session ID:', this.sessionId);
    }

    generateSessionId() {
        // Generate a simple unique session ID
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    setupBeforeUnload() {
        // Handle page close/refresh - send reset request
        window.addEventListener('beforeunload', () => {
            // Send synchronous reset if we're generating
            if (!this.isIdle) {
                // Use Blob to set correct Content-Type for JSON
                const blob = new Blob(
                    [JSON.stringify({ session_id: this.sessionId })],
                    { type: 'application/json' }
                );
                navigator.sendBeacon('/api/reset', blob);
                console.log('Sent reset beacon on page unload');
            }
        });

        // Also handle visibility change (tab hidden, mobile app switch)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.isIdle && this.isRunning) {
                // User switched away while generating - they might not come back
                // Note: Don't reset immediately, let the frame consumption monitor handle it
                console.log('Tab hidden while generating - consumption monitor will auto-reset if needed');
            }
        });
    }

    initThreeJS() {
        // Get canvas
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);  // White background

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        // Add lights - bright and soft
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(5, 8, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);

        // Add ground plane - light gray, very large
        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Add infinite-looking grid - very large grid
        const gridHelper = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();

        // Raycast helpers to map mouse position to ground (XZ plane)
        this.raycaster = new THREE.Raycaster();
        this.ndcMouse = new THREE.Vector2();
        // Match the demo's "ground" concept for trajectory selection.
        // (We project to y=0; the y used for trajectory points comes from `initialTaskY`.)
        this.groundSelectionPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

        // Visualize drawn trajectory points (small spheres)
        this.trajPointsGroup = new THREE.Group();
        this.scene.add(this.trajPointsGroup);
        const trajPointGeometry = new THREE.SphereGeometry(0.04, 14, 14);
        const trajPointMaterial = new THREE.MeshStandardMaterial({
            color: 0xff3b30,
            metalness: 0.2,
            roughness: 0.4,
            emissive: 0x330000
        });
        this.trajPointGeometry = trajPointGeometry;
        this.trajPointMaterial = trajPointMaterial;

        // Canvas pointer interactions: drag on ground to create waypoints
        canvas.addEventListener('pointerdown', (e) => this.onCanvasPointerDown(e));
        canvas.addEventListener('pointermove', (e) => this.onCanvasPointerMove(e));
        canvas.addEventListener('pointerup', (e) => this.onCanvasPointerUp(e));
        canvas.addEventListener('pointercancel', (e) => this.onCanvasPointerUp(e));

        // Listen for user interaction - record time
        const updateInteractionTime = () => {
            this.lastUserInteraction = Date.now();
        };
        canvas.addEventListener('mousedown', updateInteractionTime);
        canvas.addEventListener('wheel', updateInteractionTime);
        canvas.addEventListener('touchstart', updateInteractionTime);

        // Create skeleton
        this.skeleton = new Skeleton3D(this.scene);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start render loop
        this.animate();
    }

    initUI() {
        // Get UI elements
        this.motionText = document.getElementById('motionText');
        this.historyLength = document.getElementById('historyLength');
        this.denoiseSteps = document.getElementById('denoiseSteps');
        this.smoothingAlpha = document.getElementById('smoothingAlpha');
        this.smoothingValue = document.getElementById('smoothingValue');
        this.currentSmoothing = document.getElementById('currentSmoothing');
        this.currentSteps = document.getElementById('currentSteps');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.pauseResumeBtn = document.getElementById('pauseResumeBtn');
        this.statusEl = document.getElementById('status');
        this.bufferSizeEl = document.getElementById('bufferSize');
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.conflictWarning = document.getElementById('conflictWarning');
        this.forceTakeoverBtn = document.getElementById('forceTakeoverBtn');
        this.cancelTakeoverBtn = document.getElementById('cancelTakeoverBtn');
        this.trajectoryWaypoints = document.getElementById('trajectoryWaypoints');
        this.updateTrajBtn = document.getElementById('updateTrajBtn');
        this.clearTrajBtn = document.getElementById('clearTrajBtn');

        // Track state
        this.isPaused = false;
        this.isIdle = true;
        this.isProcessing = false;  // Prevent concurrent API calls
        this.pendingStartRequest = null;  // Store pending start request data

        // Attach event listeners
        this.startResetBtn.addEventListener('click', () => this.toggleStartReset());
        this.updateBtn.addEventListener('click', () => this.updateText());
        this.pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        this.forceTakeoverBtn.addEventListener('click', () => this.handleForceTakeover());
        this.cancelTakeoverBtn.addEventListener('click', () => this.handleCancelTakeover());
        if (this.updateTrajBtn) this.updateTrajBtn.addEventListener('click', () => this.updateTrajectory());
        if (this.clearTrajBtn) this.clearTrajBtn.addEventListener('click', () => this.clearTrajectory());

        // Update smoothing value display when slider changes
        this.smoothingAlpha.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value).toFixed(2);
            this.smoothingValue.textContent = value;
        });
    }

    async toggleStartReset() {
        if (this.isProcessing) return;  // Prevent concurrent operations

        if (this.isIdle) {
            // Currently idle, so start
            await this.startGeneration();
        } else {
            // Currently running/paused, so reset
            await this.reset();
        }
    }

    async startGeneration(force = false) {
        if (this.isProcessing) return;  // Prevent concurrent operations

        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }

        const historyLength = parseInt(this.historyLength.value) || 30;
        if (historyLength < 10 || historyLength > 200) {
            alert('History length must be between 10 and 200');
            return;
        }

        const denoiseSteps = parseInt(this.denoiseSteps.value) || 10;
        if (denoiseSteps < 5 || denoiseSteps > 50) {
            alert('Denoising steps must be between 5 and 50');
            return;
        }
        if (denoiseSteps % 5 !== 0) {
            alert('Denoising steps must be a multiple of 5 (e.g., 5, 10, 15, 20...)');
            return;
        }

        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);

        this.isProcessing = true;
        this.statusEl.textContent = 'Initializing...';

        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps,
                    force: force
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.isIdle = false;
                // Reset "initial task y" capture; it will be updated after first frame.
                this.taskYCaptured = false;
                this.initialTaskY = this.currentRootPos.y;
                this.frameCount = 0;
                this.motionFrameCount = 0;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = performance.now();
                this.isFetchingFrame = false;
                this.consecutiveWaiting = 0;
                this.startResetBtn.textContent = 'Reset';
                this.startResetBtn.classList.remove('btn-primary');
                this.startResetBtn.classList.add('btn-danger');
                this.updateBtn.disabled = false;
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();

                // If there's trajectory in textarea (or from drag), apply it immediately after start.
                // Otherwise users need to click "Update Trajectory" manually.
                const initTraj = this.parseWaypointsFromTextarea();
                if (initTraj && initTraj.length > 0) {
                    this.drawnWaypoints = initTraj;
                    this.syncTrajectorySpheresFromWaypoints();
                    this.pushTrajectoryToBackend(initTraj, true);
                }
            } else if (response.status === 409 && data.conflict) {
                // Another session is running, show warning UI
                this.statusEl.textContent = 'Conflict - Another user is generating';
                this.conflictWarning.style.display = 'block';

                // Store request data for later
                this.pendingStartRequest = {
                    text: text,
                    history_length: historyLength
                };

                return;
            } else {
                // Other errors
                alert('Error: ' + data.message);
                this.statusEl.textContent = 'Idle';
                this.isIdle = true;
                this.isRunning = false;
                this.isPaused = false;
            }
        } catch (error) {
            console.error('Error starting generation:', error);
            alert('Failed to start generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            // Keep idle state on error
            this.isIdle = true;
            this.isRunning = false;
            this.isPaused = false;
        } finally {
            this.isProcessing = false;
        }
    }

    async updateText() {
        if (this.isProcessing) return;  // Prevent concurrent operations

        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }

        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log('Text updated:', text);
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error updating text:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    /** Parse trajectory waypoints from textarea: one line = one point, "x z" or "x y z". */
    parseWaypointsFromTextarea() {
        const text = (this.trajectoryWaypoints && this.trajectoryWaypoints.value)
            ? this.trajectoryWaypoints.value.trim()
            : '';
        if (!text) return null;
        const lines = text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
        if (lines.length === 0) return null;
        const points = [];
        for (const line of lines) {
            const parts = line.split(/\s+/).map(parseFloat);
            if (parts.length >= 2 && parts.every((n) => !Number.isNaN(n))) {
                points.push(
                    parts.length >= 3
                        ? [parts[0], parts[1], parts[2]]
                        : [parts[0], this.initialTaskY ?? 0, parts[1]]
                );
            }
        }
        return points.length > 0 ? points : null;
    }

    async updateTrajectory() {
        if (this.isProcessing) return;
        const waypoints = this.parseWaypointsFromTextarea();
        if (!waypoints) {
            alert('Enter at least one waypoint (x z or x y z per line).');
            return;
        }
        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId, waypoints })
            });
            const data = await response.json();
            if (data.status === 'success') {
                console.log('Trajectory updated:', waypoints.length, 'waypoints');
                this.drawnWaypoints = waypoints;
                this.syncTrajectorySpheresFromWaypoints();
            } else {
                alert('Error: ' + (data.message || 'Failed to update trajectory'));
            }
        } catch (e) {
            console.error('Update trajectory error:', e);
            alert('Failed to update trajectory');
        } finally {
            this.isProcessing = false;
        }
    }

    async clearTrajectory() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId, waypoints: null })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.clearDrawnTrajectoryUI();
                this.drawnWaypoints = [];
                console.log('Trajectory cleared');
            }
        } catch (e) {
            console.error('Clear trajectory error:', e);
        } finally {
            this.isProcessing = false;
        }
    }

    async togglePauseResume() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        if (this.isPaused) {
            // Currently paused, so resume
            await this.resumeGeneration();
        } else {
            // Currently running, so pause
            await this.pauseGeneration();
        }
    }

    async pauseGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/pause', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = true;
                this.pauseResumeBtn.textContent = 'Resume';
                this.pauseResumeBtn.classList.remove('btn-warning');
                this.pauseResumeBtn.classList.add('btn-success');
                this.updateBtn.disabled = true;
                this.statusEl.textContent = 'Paused';
                console.log('Generation paused (state preserved)');
            }
        } catch (error) {
            console.error('Error pausing generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async resumeGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.updateBtn.disabled = false;
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
                console.log('Generation resumed');
            }
        } catch (error) {
            console.error('Error resuming generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async reset() {
        if (this.isProcessing) return;  // Prevent concurrent operations

        const historyLength = parseInt(this.historyLength.value) || 30;
        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);
        const denoiseSteps = parseInt(this.denoiseSteps.value) || 10;

        this.isProcessing = true;
        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = false;
                this.isIdle = true;
                this.frameCount = 0;
                this.motionFrameCount = 0;
                this.motionFpsCounter = 0;
                this.isFetchingFrame = false;
                this.consecutiveWaiting = 0;
                this.startResetBtn.textContent = 'Start';
                this.startResetBtn.classList.remove('btn-danger');
                this.startResetBtn.classList.add('btn-primary');
                this.updateBtn.disabled = true;
                this.pauseResumeBtn.disabled = true;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.statusEl.textContent = 'Idle';
                this.bufferSizeEl.textContent = '0 / 4';
                this.frameCountEl.textContent = '0';
                this.fpsEl.textContent = '0';

                // Clear trail
                if (this.skeleton) {
                    this.skeleton.clearTrail();
                }

                // Clear drawn trajectory points (mouse-drawn)
                this.clearDrawnTrajectoryUI();

                console.log('Reset complete - all state cleared');
            }
        } catch (error) {
            console.error('Error resetting:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async handleForceTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';

        if (!this.pendingStartRequest) return;

        // Retry with force=true
        this.isProcessing = false;
        await this.startGeneration(true);

        this.pendingStartRequest = null;
    }

    handleCancelTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        this.statusEl.textContent = 'Idle';
        this.isProcessing = false;
        this.pendingStartRequest = null;
    }

    startFrameLoop() {
        const now = performance.now();
        this.lastFetchTime = now;
        this.nextFetchTime = now + this.frameInterval;
        this.fetchFrame();
    }

    fetchFrame() {
        if (!this.isRunning) return;

        const now = performance.now();

        // Check if it's time to fetch next frame AND we're not already fetching
        if (now >= this.nextFetchTime && !this.isFetchingFrame) {
            // Schedule next fetch (maintain fixed rate regardless of delays)
            this.nextFetchTime += this.frameInterval;

            // If we've fallen behind, catch up
            if (this.nextFetchTime < now) {
                this.nextFetchTime = now + this.frameInterval;
            }

            // Mark as fetching to prevent concurrent requests
            this.isFetchingFrame = true;

            fetch(`/api/get_frame?session_id=${this.sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.skeleton.updatePose(data.joints);
                        this.frameCount++;
                        this.frameCountEl.textContent = this.frameCount;

                        // Update motion FPS counter (only when frame consumed)
                        this.motionFrameCount++;
                        this.motionFpsCounter++;

                        // Update current root position
                        this.currentRootPos.set(
                            data.joints[0][0],
                            data.joints[0][1],
                            data.joints[0][2]
                        );

                        // Capture "initial task y" once after generation starts.
                        if (!this.taskYCaptured) {
                            this.initialTaskY = this.currentRootPos.y;
                            this.taskYCaptured = true;
                        }

                        // Auto-follow (if user hasn't interacted for a while)
                        this.updateAutoFollow();

                        // Reset waiting counter on success
                        this.consecutiveWaiting = 0;
                    } else if (data.status === 'waiting') {
                        // No frame available, slow down requests if this happens repeatedly
                        this.consecutiveWaiting++;

                        // If buffer is consistently empty, back off a bit
                        if (this.consecutiveWaiting > 5) {
                            // Add a small delay to reduce server load
                            this.nextFetchTime = now + this.frameInterval * 1.5;
                            this.consecutiveWaiting = 0;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching frame:', error);
                })
                .finally(() => {
                    // Always mark as done fetching
                    this.isFetchingFrame = false;
                });
        }

        // Use requestAnimationFrame for continuous checking
        requestAnimationFrame(() => this.fetchFrame());
    }

    updateAutoFollow() {
        const timeSinceInteraction = Date.now() - this.lastUserInteraction;

        // Auto-follow if user hasn't interacted for more than 3 seconds
        if (timeSinceInteraction > this.autoFollowDelay) {
            // Calculate camera offset relative to current target
            const currentOffset = new THREE.Vector3().subVectors(
                this.camera.position,
                this.controls.target
            );

            // New target position (character position, waist height)
            const newTarget = this.currentRootPos.clone();
            newTarget.y = 1.0;

            // Calculate new camera position (maintain relative offset)
            const newCameraPos = newTarget.clone().add(currentOffset);

            // Smooth interpolation follow (increased lerp factor for more obvious following)
            // 0.2 = more aggressive following, 0.05 = gentle following
            this.controls.target.lerp(newTarget, 0.2);
            this.camera.position.lerp(newCameraPos, 0.2);

            // Debug log (comment out in production)
            // console.log('Auto-follow active, tracking:', newTarget);
        }
    }

    getGroundPointFromEvent(e) {
        // Project pointer to NDC and raycast onto a y=0 plane.
        // Returned y is NOT the intersection y; it uses `initialTaskY` (for xz-only trajectory input).
        if (!this.renderer || !this.camera || !this.raycaster || !this.groundSelectionPlane) return null;

        const rect = this.renderer.domElement.getBoundingClientRect();
        const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

        this.ndcMouse.set(ndcX, ndcY);
        this.raycaster.setFromCamera(this.ndcMouse, this.camera);

        const hitPoint = new THREE.Vector3();
        const hit = this.raycaster.ray.intersectPlane(this.groundSelectionPlane, hitPoint);
        if (!hit) return null;

        return [hitPoint.x, this.initialTaskY, hitPoint.z];
    }

    clearTrajectorySpheres() {
        if (!this.trajPointsGroup) return;
        for (const mesh of this.trajPointSpheres) {
            this.trajPointsGroup.remove(mesh);
        }
        this.trajPointSpheres = [];
    }

    syncTrajectorySpheresFromWaypoints(waypoints) {
        this.clearTrajectorySpheres();
        if (!waypoints || waypoints.length === 0) return;

        for (const p of waypoints) {
            const mesh = new THREE.Mesh(this.trajPointGeometry, this.trajPointMaterial);
            mesh.position.set(p[0], p[1], p[2]);
            this.trajPointsGroup.add(mesh);
            this.trajPointSpheres.push(mesh);
        }
    }

    syncTextareaFromWaypoints(waypoints) {
        if (!this.trajectoryWaypoints) return;
        if (!waypoints || waypoints.length === 0) {
            this.trajectoryWaypoints.value = '';
            return;
        }
        // Display as x z only; y will be filled during parsing using `initialTaskY`.
        this.trajectoryWaypoints.value = waypoints
            .map((p) => `${Number(p[0]).toFixed(3)} ${Number(p[2]).toFixed(3)}`)
            .join('\n');
    }

    async pushTrajectoryToBackend(waypoints, force = false) {
        if (this.isIdle) return;

        const now = performance.now();
        if (!force && now - this.lastTrajectoryPushTime < this.trajectoryPushThrottleMs) return;

        if (this.trajectoryPushInFlight) {
            this.pendingTrajectoryPush = true;
            return;
        }

        this.trajectoryPushInFlight = true;
        this.lastTrajectoryPushTime = now;
        try {
            await fetch('/api/update_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    waypoints: waypoints && waypoints.length > 0 ? waypoints : null,
                    mode: 'replace_future'
                })
            });
        } catch (err) {
            console.error('pushTrajectoryToBackend failed:', err);
        } finally {
            this.trajectoryPushInFlight = false;
            if (this.pendingTrajectoryPush) {
                this.pendingTrajectoryPush = false;
                this.pushTrajectoryToBackend(this.drawnWaypoints, true);
            }
        }
    }

    clearDrawnTrajectoryUI() {
        this.drawnWaypoints = [];
        this.syncTextareaFromWaypoints([]);
        this.clearTrajectorySpheres();
    }

    addWaypointFromPoint(xyz) {
        if (!xyz) return false;
        const last = this.drawnWaypoints.length > 0 ? this.drawnWaypoints[this.drawnWaypoints.length - 1] : null;
        if (last) {
            const dx = xyz[0] - last[0];
            const dz = xyz[2] - last[2];
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < this.drawnWaypointMinDist) return false;
        }
        this.drawnWaypoints.push(xyz);
        return true;
    }

    onCanvasPointerDown(e) {
        if (e.button !== 0) return;
        if (!this.controls) return;

        this.controls.enabled = false;
        this.isDrawingTrajectory = true;

        // Start a new stroke.
        this.drawnWaypoints = [];
        this.pendingTrajectoryPush = false;
        this.trajectoryPushInFlight = false;
        this.clearTrajectorySpheres();
        this.syncTextareaFromWaypoints([]);

        const p = this.getGroundPointFromEvent(e);
        const added = this.addWaypointFromPoint(p);
        if (added) {
            this.syncTextareaFromWaypoints(this.drawnWaypoints);
            this.syncTrajectorySpheresFromWaypoints(this.drawnWaypoints);
            this.pushTrajectoryToBackend(this.drawnWaypoints, true);
        }

        this.lastUserInteraction = Date.now();
    }

    onCanvasPointerMove(e) {
        if (!this.isDrawingTrajectory) return;
        const p = this.getGroundPointFromEvent(e);
        const added = this.addWaypointFromPoint(p);
        if (!added) return;

        // Keep UI responsive; backend updates are throttled.
        this.syncTextareaFromWaypoints(this.drawnWaypoints);
        this.syncTrajectorySpheresFromWaypoints(this.drawnWaypoints);
        this.pushTrajectoryToBackend(this.drawnWaypoints, false);

        this.lastUserInteraction = Date.now();
    }

    onCanvasPointerUp(_e) {
        if (!this.isDrawingTrajectory) return;
        this.isDrawingTrajectory = false;
        this.controls.enabled = true;

        if (this.drawnWaypoints && this.drawnWaypoints.length > 0) {
            this.pushTrajectoryToBackend(this.drawnWaypoints, true);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${this.sessionId}`);
            const data = await response.json();

            if (data.initialized) {
                this.bufferSizeEl.textContent = `${data.buffer_size} / ${data.target_size}`;

                // Update current smoothing display
                if (data.smoothing_alpha !== undefined) {
                    this.currentSmoothing.textContent = data.smoothing_alpha.toFixed(2);
                }

                // Update current denoising steps display
                if (data.denoise_steps !== undefined) {
                    this.currentSteps.textContent = data.denoise_steps;
                }
            }

            // Update motion FPS (frame consumption rate)
            const now = performance.now();
            if (now - this.motionFpsUpdateTime > 1000) {
                this.fpsEl.textContent = this.motionFpsCounter;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = now;
            }
        } catch (error) {
            // Silently fail for status updates
        }

        // Update status every 500ms
        setTimeout(() => this.updateStatus(), 500);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

        // Render scene
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.app = new MotionApp();
});
