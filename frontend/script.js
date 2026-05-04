let camOn = false;
let kpOn = true;

let goodCount = 0, badCount = 0, totalCount = 0;
let goodStreak = 0, badStreak = 0;
let lastBadTime = null;
let fpsHistory = [];
let lastStatTime = performance.now();

let alertActive = false;
let lastBeepTime = 0;
const BAD_THRESHOLD_SEC = 10;
const BEEP_INTERVAL_MS = 3000;

const video = document.getElementById("video");
const overlay = document.getElementById("cameraOffOverlay");
const alertBanner = document.getElementById("alertBanner");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctx = overlayCanvas.getContext("2d");
let processInterval;
let isProcessing = false;

function beep() {
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = actx.createOscillator();
    const gain = actx.createGain();
    oscillator.connect(gain);
    gain.connect(actx.destination);
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(880, actx.currentTime);
    gain.gain.setValueAtTime(0.4, actx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.4);
    oscillator.start(actx.currentTime);
    oscillator.stop(actx.currentTime + 0.4);
}

function startCam() {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            video.srcObject = stream;
            video.style.display = "block";
            overlayCanvas.style.display = "block";
            overlay.style.display = "none";
            video.onloadedmetadata = () => {
                overlayCanvas.width = video.videoWidth;
                overlayCanvas.height = video.videoHeight;
                startProcessingLoop();
            };
        })
        .catch(err => {
            console.error("Camera error:", err);
            camOn = false;
            document.getElementById("btnCamera").dataset.active = "false";
            alert("Unable to access camera. Please ensure permissions are granted.");
        });
}

function stopCam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    video.style.display = "none";
    overlayCanvas.style.display = "none";
    overlay.style.display = "flex";
    stopProcessingLoop();
}

function toggleCamera() {
    camOn = !camOn;
    document.getElementById("btnCamera").dataset.active = camOn ? "true" : "false";
    if (camOn) startCam(); else stopCam();
}

function toggleKP() {
    kpOn = !kpOn;
    document.getElementById("btnKP").dataset.active = kpOn ? "true" : "false";
    if (!kpOn) {
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
}

function setModel(el) {
    document.querySelectorAll(".model-btn").forEach(b => b.classList.remove("active"));
    el.classList.add("active");
    fetch(`${window.BACKEND_URL}/set_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: el.dataset.model })
    }).catch(err => console.error("setModel failed:", err));
}

function updateStats(d) {
    const now = performance.now();
    const delta = (now - lastStatTime) / 1000;
    lastStatTime = now;
    if (delta > 0 && d.status !== "none") {
        fpsHistory.push(1 / delta);
        if (fpsHistory.length > 10) fpsHistory.shift();
    }
    const fps = fpsHistory.length
        ? (fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length).toFixed(1)
        : "—";
    document.getElementById("fpsVal").textContent = fps;
    document.getElementById("fpsBadge").textContent = fps + " FPS";

    const label = document.getElementById("label");
    if (d.status === "good") {
        label.textContent = "GOOD";
        label.className = "posture-label good";
    } else if (d.status === "bad") {
        label.textContent = "BAD";
        label.className = "posture-label bad";
    } else {
        label.textContent = "—";
        label.className = "posture-label none";
    }

    const pct = Math.round((d.prob || 0) * 100);
    const fill = document.getElementById("fill");
    fill.style.width = pct + "%";
    fill.style.background = d.status === "bad" ? "#ff4d4d" : "#7fff6e";
    document.getElementById("confPct").textContent = pct + "%";
    document.getElementById("confVal").textContent = pct + "%";

    document.getElementById("side").textContent =
        d.side === "left" ? "LEFT" :
            d.side === "right" ? "RIGHT" : "—";

    const badDuration = d.bad_duration || 0;
    document.getElementById("badDuration").textContent =
        d.status === "bad" ? badDuration.toFixed(1) + "s" : "—";

    if (d.status === "bad" && badDuration >= BAD_THRESHOLD_SEC) {
        if (!alertActive) {
            alertActive = true;
            alertBanner.classList.add("visible");
        }
        const nowMs = Date.now();
        if (nowMs - lastBeepTime > BEEP_INTERVAL_MS) {
            beep();
            lastBeepTime = nowMs;
        }
    } else {
        if (alertActive) {
            alertActive = false;
            alertBanner.classList.remove("visible");
        }
    }

    if (d.status !== "none") {
        totalCount++;
        document.getElementById("detections").textContent = totalCount;

        if (d.status === "good") {
            goodCount++;
            goodStreak++;
            badStreak = 0;
        } else {
            badCount++;
            badStreak++;
            goodStreak = 0;
            lastBadTime = new Date();
        }

        document.getElementById("goodStreak").textContent = goodStreak + " frames";
        document.getElementById("badStreak").textContent = badStreak + " frames";

        const pctGood = totalCount
            ? Math.round((goodCount / totalCount) * 100)
            : 0;
        document.getElementById("sessionGood").textContent = pctGood + "%";

        document.getElementById("lastCorrection").textContent =
            lastBadTime ? lastBadTime.toLocaleTimeString() : "—";
    }
}

function drawKeypoints(keypoints) {
    if (!kpOn) return;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!keypoints) return;

    // keypoints are normalized [0, 1]
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    const pt = (kp) => ({ x: kp.x * w, y: kp.y * h });

    const ear = pt(keypoints.ear);
    const shoulder = pt(keypoints.shoulder);
    const hip = pt(keypoints.hip);

    ctx.fillStyle = "#00e5ff";
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 3;

    [ear, shoulder, hip].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
        ctx.fill();
    });

    ctx.beginPath();
    ctx.moveTo(ear.x, ear.y);
    ctx.lineTo(shoulder.x, shoulder.y);
    ctx.lineTo(hip.x, hip.y);
    ctx.stroke();
}

async function processFrame() {
    if (!camOn || !video.videoWidth || isProcessing) return;
    isProcessing = true;

    // Use a temporary canvas to get base64
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tctx = tempCanvas.getContext("2d");
    
    // Draw current frame and flip horizontally if desired (MediaPipe expects a standard orientation, we send as is)
    tctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    // Compress for network constraints (0.6 balances quality vs bandwidth)
    const dataURL = tempCanvas.toDataURL("image/jpeg", 0.6);

    try {
        const response = await fetch(`${window.BACKEND_URL}/process_frame`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataURL })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        updateStats(data);
        drawKeypoints(data.keypoints);
    } catch (err) {
        console.error("Frame processing failed:", err);
    } finally {
        isProcessing = false;
    }
}

function startProcessingLoop() {
    // Poll every 300ms (approx 3 FPS) - safe for Render free tier
    processInterval = setInterval(processFrame, 300);
}

function stopProcessingLoop() {
    if (processInterval) clearInterval(processInterval);
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function toggleMobileMenu() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("open");
}

stopCam();