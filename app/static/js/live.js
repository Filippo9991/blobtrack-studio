// BlobTrack Studio — Live cam (Pass 3).
//
// La webcam la fornisce il browser (getUserMedia). Per ogni frame: lo disegno
// su un canvas ridotto, lo codifico in JPEG e lo POSTo a /live/frame insieme al
// config corrente; il server risponde col frame elaborato che mostro in <img>.
// Una sola richiesta in volo alla volta (niente accodamento): l'FPS si adatta
// alla latenza del server.

(function () {
  var form = document.getElementById("live-form");
  if (!form) return;

  var startBtn = document.getElementById("cam-start");
  var stopBtn = document.getElementById("cam-stop");
  var snapBtn = document.getElementById("snapshot-btn");
  var video = document.getElementById("cam-video");
  var output = document.getElementById("live-output");
  var placeholder = document.getElementById("live-placeholder");
  var hud = document.getElementById("live-hud");
  var fpsEl = document.getElementById("live-fps");
  var snapshotField = document.getElementById("snapshot_raw");

  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  var CAPTURE_WIDTH = 640; // ridimensiona prima di inviare: leggero e veloce

  var stream = null;
  var running = false;
  var inFlight = false;
  var objectUrl = null;
  var frames = 0;
  var fpsTimer = null;

  // Legge i campi del form e costruisce il dict di config per il motore.
  // Solo le checkbox diventano booleani; il resto resta stringa (pydantic coerce).
  var SKIP = { csrf_token: 1, snapshot_raw: 1, preset_name: 1, submit: 1, action: 1 };
  function readConfig() {
    var cfg = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || SKIP[el.name]) return;
      cfg[el.name] = el.type === "checkbox" ? el.checked : el.value;
    });
    return cfg;
  }

  // Disegna il frame corrente della webcam sul canvas ridotto -> data URL JPEG.
  function captureFrame() {
    var vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.round((vh / vw) * CAPTURE_WIDTH);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7);
  }

  function loop() {
    if (!running || inFlight) return;
    var frame = captureFrame();
    if (!frame) { requestAnimationFrame(loop); return; }

    inFlight = true;
    fetch("/live/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frame, config: readConfig() }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.blob();
      })
      .then(function (blob) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(blob);
        output.src = objectUrl;
        output.style.display = "block";
        placeholder.style.display = "none";
        frames++;
      })
      .catch(function () { /* salta il frame e prosegui */ })
      .finally(function () {
        inFlight = false;
        if (running) requestAnimationFrame(loop);
      });
  }

  function start() {
    if (running || !navigator.mediaDevices) {
      if (!navigator.mediaDevices) placeholder.textContent = "Webcam non supportata dal browser.";
      return;
    }
    placeholder.textContent = "Avvio camera…";
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then(function (s) {
        stream = s;
        video.srcObject = s;
        return video.play();
      })
      .then(function () {
        running = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        snapBtn.disabled = false;
        hud.hidden = false;
        frames = 0;
        fpsTimer = setInterval(function () { fpsEl.textContent = frames; frames = 0; }, 1000);
        requestAnimationFrame(loop);
      })
      .catch(function (err) {
        placeholder.textContent = "Camera non disponibile: " + (err && err.message ? err.message : err);
      });
  }

  function stop() {
    running = false;
    if (fpsTimer) { clearInterval(fpsTimer); fpsTimer = null; }
    fpsEl.textContent = "0";
    hud.hidden = true;
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    video.srcObject = null;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    output.removeAttribute("src");
    output.style.display = "none";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    snapBtn.disabled = true;
    placeholder.style.display = "block";
    placeholder.textContent = "Camera ferma. Avvia di nuovo per riprendere.";
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);

  // Snapshot: metto il frame grezzo corrente nel campo nascosto e lascio fare il
  // submit classico (CSRF protetto). Il server rielabora e salva in galleria.
  form.addEventListener("submit", function (e) {
    var action = document.activeElement && document.activeElement.value;
    if (action !== "save_snapshot") return;
    if (!running) { e.preventDefault(); return; }
    snapshotField.value = captureFrame() || "";
  });

  window.addEventListener("beforeunload", function () {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
  });
})();
