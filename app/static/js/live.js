// BlobTrack Studio — Live cam (Pass 3 + Pass 6).
//
// La webcam la fornisce il browser (getUserMedia). Per ogni frame: lo disegno
// su un canvas ridotto, lo codifico in JPEG e lo mando al server insieme al
// config corrente; il server risponde col frame elaborato che mostro in <img>.
//
// Trasporto: prova il WebSocket (/live/ws, un handshake solo → FPS più alti);
// se non disponibile ripiega sul polling HTTP (/live/frame). In entrambi i casi
// UNA sola richiesta in volo alla volta: l'FPS si adatta alla latenza.
//
// Stato: a ogni "Avvia" viene generato uno stream id; sul server quel id ha un
// motore dedicato con stato persistente (scie, ID di tracking stabili).
//
// Microfono: se una modulazione audio è attiva, WebAudio calcola il livello
// (RMS, attacco immediato/rilascio dolce) e lo invia col frame.

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
  var transportEl = document.getElementById("live-transport");
  var snapshotField = document.getElementById("snapshot_raw");

  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  var CAPTURE_WIDTH = 640; // ridimensiona prima di inviare: leggero e veloce
  var IN_FLIGHT_TIMEOUT = 6000; // ms senza risposta -> il frame si considera perso

  var stream = null;
  var running = false;
  var inFlight = false;
  var sentAt = 0;
  var objectUrl = null;
  var frames = 0;
  var fpsTimer = null;
  var streamId = null;

  var ws = null; // WebSocket attivo (null = polling HTTP)

  // --- Microfono (WebAudio) --------------------------------------------------
  var audioCtx = null;
  var analyser = null;
  var audioData = null;
  var micStream = null;
  var micRequested = false; // evita di richiedere il permesso a raffica se negato
  var audioLevel = 0;

  var MOD_FIELDS = ["audio_modulate_size", "audio_modulate_thickness", "audio_modulate_glow"];
  function micWanted() {
    return MOD_FIELDS.some(function (name) {
      var el = form.elements[name];
      return el && el.checked;
    });
  }

  function startMic() {
    if (micRequested || !navigator.mediaDevices) return;
    micRequested = true;
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(function (s) {
        micStream = s;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        audioCtx.createMediaStreamSource(s).connect(analyser);
        audioData = new Uint8Array(analyser.frequencyBinCount);
      })
      .catch(function () { /* permesso negato: si continua senza reattività */ });
  }

  function stopMic() {
    if (micStream) { micStream.getTracks().forEach(function (t) { t.stop(); }); micStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    analyser = null;
    micRequested = false;
    audioLevel = 0;
  }

  function readAudioLevel() {
    if (!analyser) return 0;
    analyser.getByteTimeDomainData(audioData);
    var sum = 0;
    for (var i = 0; i < audioData.length; i++) {
      var v = (audioData[i] - 128) / 128;
      sum += v * v;
    }
    var rms = Math.sqrt(sum / audioData.length);
    var level = Math.min(1, rms * 4); // gain: parlato/musica ~0.05–0.3 RMS
    // attacco immediato, rilascio dolce: i colpi si vedono, senza sfarfallio
    audioLevel = Math.max(level, audioLevel * 0.85);
    return Math.round(audioLevel * 1000) / 1000;
  }

  // --- Config e cattura -------------------------------------------------------
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

  function showResult(blob) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(blob);
    output.src = objectUrl;
    output.style.display = "block";
    placeholder.style.display = "none";
    frames++;
  }

  // --- Trasporto: WebSocket con fallback HTTP --------------------------------
  function connectWS() {
    return new Promise(function (resolve) {
      var sock;
      try {
        var proto = location.protocol === "https:" ? "wss://" : "ws://";
        sock = new WebSocket(proto + location.host + "/live/ws");
      } catch (e) { resolve(null); return; }
      sock.binaryType = "blob";
      var settled = false;
      sock.onopen = function () { if (!settled) { settled = true; resolve(sock); } };
      sock.onerror = function () { if (!settled) { settled = true; resolve(null); } };
      sock.onclose = function () {
        if (!settled) { settled = true; resolve(null); }
        if (ws === sock) { // caduto in corsa: si prosegue in polling
          ws = null;
          inFlight = false;
          transportEl.textContent = "HTTP";
        }
      };
      sock.onmessage = function (ev) {
        inFlight = false;
        if (ev.data instanceof Blob) showResult(ev.data);
        // messaggi di testo ("{}") = frame saltato dal server: si prosegue
        if (running) requestAnimationFrame(loop);
      };
    });
  }

  function buildPayload(frame) {
    return {
      frame: frame,
      config: readConfig(),
      stream: streamId,
      audio_level: micWanted() ? readAudioLevel() : 0,
    };
  }

  function loop() {
    if (!running) return;
    if (inFlight) {
      // watchdog: una risposta persa non deve congelare lo stream
      if (Date.now() - sentAt > IN_FLIGHT_TIMEOUT) inFlight = false;
      else return;
    }
    var frame = captureFrame();
    if (!frame) { requestAnimationFrame(loop); return; }

    // il mic serve? richiedilo appena l'utente attiva una modulazione
    if (micWanted() && !micRequested) startMic();

    inFlight = true;
    sentAt = Date.now();

    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify(buildPayload(frame))); // risposta in ws.onmessage
      // rete di sicurezza: se la risposta si perde, il watchdog rilancia il loop
      setTimeout(loop, IN_FLIGHT_TIMEOUT + 50);
      return;
    }

    fetch("/live/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload(frame)),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.blob();
      })
      .then(showResult)
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
        // nuovo stream = stato di tracking pulito sul server
        streamId = (window.crypto && crypto.randomUUID)
          ? crypto.randomUUID()
          : Date.now().toString(36) + Math.random().toString(36).slice(2);
        return connectWS();
      })
      .then(function (sock) {
        ws = sock;
        transportEl.textContent = ws ? "WS" : "HTTP";
        running = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        snapBtn.disabled = false;
        hud.hidden = false;
        frames = 0;
        fpsTimer = setInterval(function () { fpsEl.textContent = frames; frames = 0; }, 1000);
        if (micWanted()) startMic();
        requestAnimationFrame(loop);
      })
      .catch(function (err) {
        placeholder.textContent = "Camera non disponibile: " + (err && err.message ? err.message : err);
      });
  }

  function stop() {
    running = false;
    inFlight = false;
    if (ws) { try { ws.close(); } catch (e) { /* già chiuso */ } ws = null; }
    stopMic();
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
    if (micStream) micStream.getTracks().forEach(function (t) { t.stop(); });
  });
})();
