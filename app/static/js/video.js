// BlobTrack Studio — pagina Video: anteprima + elaborazione asincrona.
//
// 1. ANTEPRIMA SORGENTE: alla scelta del file, il video appare subito nel
//    pannello (object URL, nessun upload).
// 2. ANTEPRIMA FRAME: "Anteprima frame" cattura il fotogramma corrente del
//    player, lo manda a /live/frame (elaborazione singola, stateless) con i
//    parametri correnti della sidebar e mostra il risultato: si prova il look
//    PRIMA di lanciare il render completo.
// 3. RENDER ASINCRONO: il submit parte via fetch → il server risponde con un
//    job id → polling su /video/status/<id> aggiorna la barra di avanzamento
//    (il progresso arriva dal progress_callback del motore) → a fine job si
//    passa alla pagina del risultato (player + download).
// Senza JS il form fa il POST classico sincrono: stessa elaborazione, niente barra.

(function () {
  var form = document.getElementById("studio-form");
  var stage = document.getElementById("video-stage");
  var fileInput = form && form.querySelector('input[name="video"]');
  if (!form || !fileInput || !stage) return; // non è la pagina Video

  var placeholder = document.getElementById("video-placeholder");
  var resultBox = document.getElementById("result-box");
  var srcPreview = document.getElementById("src-preview");
  var previewActions = document.getElementById("preview-actions");
  var frameBtn = document.getElementById("btn-frame-preview");
  var frameWrap = document.getElementById("frame-preview-wrap");
  var frameImg = document.getElementById("frame-preview");
  var progressBox = document.getElementById("job-progress");
  var progressFill = document.getElementById("progress-fill");
  var progressLabel = document.getElementById("progress-label");
  var errorBox = document.getElementById("job-error");
  var submitBtn = form.querySelector('button[type="submit"]');

  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  var CAPTURE_WIDTH = 640;
  var objectUrl = null;
  var frameObjectUrl = null;
  var polling = null;

  function showError(msg) {
    if (!errorBox) return;
    errorBox.textContent = msg;
    errorBox.hidden = !msg;
  }

  // --- 1. Anteprima del video sorgente --------------------------------------
  fileInput.addEventListener("change", function () {
    showError("");
    if (!fileInput.files || !fileInput.files.length) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(fileInput.files[0]);
    srcPreview.src = objectUrl;
    srcPreview.hidden = false;
    previewActions.hidden = false;
    if (placeholder) placeholder.style.display = "none";
    if (resultBox) resultBox.hidden = true; // nuovo giro: via il risultato vecchio
    srcPreview.play().catch(function () { /* autoplay bloccato: ok */ });
  });

  // Codec non decodificabile dal browser (es. mp4v): l'elaborazione server
  // funziona comunque, ma anteprime e player non possono mostrarlo.
  srcPreview.addEventListener("error", function () {
    showError("Il browser non riesce a riprodurre questo file (codec non supportato): " +
              "l'elaborazione funziona comunque, ma senza anteprima.");
  });

  // --- 2. Anteprima del frame corrente elaborato -----------------------------
  var SKIP = { csrf_token: 1, video: 1, audio: 1, preset_name: 1, submit: 1, action: 1 };
  function readConfig() {
    var cfg = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || SKIP[el.name]) return;
      cfg[el.name] = el.type === "checkbox" ? el.checked : el.value;
    });
    return cfg;
  }

  function captureFrame() {
    var vw = srcPreview.videoWidth, vh = srcPreview.videoHeight;
    if (!vw || !vh) return null;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.round((vh / vw) * CAPTURE_WIDTH);
    ctx.drawImage(srcPreview, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  if (frameBtn) frameBtn.addEventListener("click", function () {
    var frame = captureFrame();
    if (!frame) { showError("Il video non è ancora pronto: attendi un istante."); return; }
    showError("");
    frameBtn.disabled = true;
    frameBtn.textContent = "Elaboro…";
    fetch("/live/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frame, config: readConfig() }),
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || "HTTP " + r.status); });
        return r.blob();
      })
      .then(function (blob) {
        if (frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
        frameObjectUrl = URL.createObjectURL(blob);
        frameImg.src = frameObjectUrl;
        frameWrap.hidden = false;
      })
      .catch(function (err) { showError("Anteprima fallita: " + err.message); })
      .finally(function () {
        frameBtn.disabled = false;
        frameBtn.textContent = "Anteprima frame";
      });
  });

  // --- 3. Render asincrono con barra di avanzamento --------------------------
  function setProgress(fraction) {
    var pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    progressFill.style.width = pct + "%";
    progressLabel.textContent = pct + "%";
  }

  function pollStatus(job) {
    fetch("/video/status/" + job)
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (s.state === "done") {
          setProgress(1);
          window.location.href = s.redirect;
          return;
        }
        if (s.state === "error") {
          progressBox.hidden = true;
          submitBtn.disabled = false;
          showError("Elaborazione fallita: " + (s.error || "errore sconosciuto"));
          return;
        }
        setProgress(s.progress || 0);
        polling = setTimeout(function () { pollStatus(job); }, 1000);
      })
      .catch(function () { // errore di rete transitorio: riprova
        polling = setTimeout(function () { pollStatus(job); }, 2000);
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    showError("");
    if (!fileInput.files || !fileInput.files.length) {
      showError("Carica un video prima di elaborare.");
      return;
    }
    submitBtn.disabled = true;
    frameWrap.hidden = true;
    progressBox.hidden = false;
    setProgress(0);

    fetch(window.location.pathname, {
      method: "POST",
      headers: { "X-Requested-With": "fetch" },
      body: new FormData(form),
    })
      .then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.error || "HTTP " + r.status);
          return d;
        });
      })
      .then(function (d) { pollStatus(d.job); })
      .catch(function (err) {
        progressBox.hidden = true;
        submitBtn.disabled = false;
        showError(err.message);
      });
  });

  window.addEventListener("beforeunload", function () {
    if (polling) clearTimeout(polling);
  });
})();
