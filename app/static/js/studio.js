// BlobTrack Studio — pagina unificata: sorgente IMMAGINE o VIDEO, un'unica UI.
//
// L'anteprima mostrata è SEMPRE quella ELABORATA da blobtrack (mai la sorgente
// grezza), e si aggiorna da sola: al caricamento, durante la riproduzione del
// video (con controlli play/pausa + timeline propri, nessun player grezzo),
// allo scrub e a ogni modifica dei parametri. Ogni frame è elaborato da
// /live/frame (singolo frame, stateless), una richiesta in volo alla volta.
//
// Azioni: "Salva creazione" salva il fotogramma corrente (immagine o frame
// video) in galleria; "Elabora e scarica video" avvia il render completo
// (tracker/scie/audio) come job asincrono con barra di avanzamento; "Salva
// preset" salva la configurazione. Senza JS i pulsanti fanno POST classici.

(function () {
  var form = document.getElementById("studio-form");
  if (!form) return;
  var fileInput = form.querySelector('input[name="source"]');
  var snapshotField = document.getElementById("snapshot_raw");

  // Elementi del pannello anteprima (assenti sulle pagine risultato/anteprima-nojs)
  var stage = document.getElementById("studio-stage");
  var placeholder = document.getElementById("studio-placeholder");
  var livePreview = document.getElementById("live-preview");
  var videoControls = document.getElementById("video-controls");
  var ppBtn = document.getElementById("pp-btn");
  var timeline = document.getElementById("timeline");
  var timeLabel = document.getElementById("time-label");
  var progressBox = document.getElementById("job-progress");
  var progressFill = document.getElementById("progress-fill");
  var progressLabel = document.getElementById("progress-label");
  var errorBox = document.getElementById("job-error");
  var srcMedia = document.getElementById("src-media");   // <video> nascosto (sorgente)
  var srcImage = document.getElementById("src-image");   // <img> nascosto (sorgente)

  var btnRender = document.getElementById("btn-render");
  var btnPreviewNojs = document.getElementById("btn-preview-nojs");
  var videoOnly = form.parentNode ? document.querySelectorAll(".video-only") : [];

  // Con JS l'anteprima immagine è live nel pannello: il bottone server-side non serve.
  if (btnPreviewNojs) btnPreviewNojs.hidden = true;

  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  var CAPTURE_WIDTH = 640;

  var srcUrl = null, previewUrl = null;
  var hasSource = false, isVideo = false;
  var inFlight = false, pending = false, paramTimer = null, polling = null;

  function showError(msg) { if (errorBox) { errorBox.textContent = msg || ""; errorBox.hidden = !msg; } }
  function setVideoOnly(show) {
    Array.prototype.forEach.call(videoOnly, function (el) { el.hidden = !show; });
    if (btnRender) btnRender.hidden = !show;
    if (videoControls) videoControls.hidden = !show;
  }

  function isVideoFile(f) {
    if (f && f.type) return f.type.indexOf("video/") === 0;
    return /\.(mp4|mov|avi|webm|mkv)$/i.test((f && f.name) || "");
  }

  // Config dai campi della sidebar (checkbox → bool, il resto stringa)
  var SKIP = { csrf_token: 1, source: 1, audio: 1, preset_name: 1, submit: 1, action: 1, snapshot_raw: 1 };
  function readConfig() {
    var cfg = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || SKIP[el.name]) return;
      cfg[el.name] = el.type === "checkbox" ? el.checked : el.value;
    });
    return cfg;
  }

  // Data URL del fotogramma GREZZO corrente (immagine o frame video), o null.
  function captureRawFrame() {
    var el, w, h;
    if (isVideo) { el = srcMedia; w = srcMedia.videoWidth; h = srcMedia.videoHeight; }
    else { el = srcImage; w = srcImage.naturalWidth; h = srcImage.naturalHeight; }
    if (!el || !w || !h) return null;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.round((h / w) * CAPTURE_WIDTH);
    ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  // Elabora il frame corrente e mostra il risultato (coalescing: l'ultimo vince).
  function refreshPreview() {
    if (!hasSource || !livePreview) return;
    if (inFlight) { pending = true; return; }
    var frame = captureRawFrame();
    if (!frame) { pending = true; return; }

    inFlight = true;
    fetch("/live/frame", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frame, config: readConfig() }),
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || "HTTP " + r.status); });
        return r.blob();
      })
      .then(function (blob) {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(blob);
        livePreview.src = previewUrl;
        livePreview.hidden = false;
        showError("");
      })
      .catch(function (err) { showError("Anteprima non disponibile: " + err.message); })
      .finally(function () {
        inFlight = false;
        if (isVideo && !srcMedia.paused && !srcMedia.ended) requestAnimationFrame(refreshPreview);
        else if (pending) { pending = false; refreshPreview(); }
      });
  }

  // --- Scelta della sorgente -------------------------------------------------
  fileInput.addEventListener("change", function () {
    showError("");
    if (!fileInput.files || !fileInput.files.length) return;
    if (srcUrl) URL.revokeObjectURL(srcUrl);
    srcUrl = URL.createObjectURL(fileInput.files[0]);
    hasSource = true;
    isVideo = isVideoFile(fileInput.files[0]);
    if (placeholder) placeholder.style.display = "none";

    setVideoOnly(isVideo);
    if (isVideo) {
      if (srcImage) srcImage.removeAttribute("src");
      srcMedia.src = srcUrl;
      srcMedia.play().catch(function () { /* autoplay bloccato: si usa play/scrub */ });
    } else {
      try { srcMedia.pause(); } catch (e) {}
      srcMedia.removeAttribute("src");
      srcImage.onload = refreshPreview;
      srcImage.src = srcUrl;
    }
  });

  // Codec non decodificabile dal browser (es. mp4v): il render server funziona,
  // ma non possiamo mostrare l'anteprima live.
  srcMedia.addEventListener("error", function () {
    if (isVideo && hasSource) showError("Il browser non riproduce questo file (codec non supportato): " +
      "il render lato server funziona, ma senza anteprima live.");
  });

  srcMedia.addEventListener("loadeddata", refreshPreview);
  srcMedia.addEventListener("play", function () { if (ppBtn) ppBtn.textContent = "⏸"; refreshPreview(); });
  srcMedia.addEventListener("pause", function () { if (ppBtn) ppBtn.textContent = "▶︎"; refreshPreview(); });
  srcMedia.addEventListener("seeked", refreshPreview);

  // --- Controlli video propri (nessun player grezzo) -------------------------
  function fmt(s) {
    if (!isFinite(s)) return "0:00";
    var m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ":" + (sec < 10 ? "0" : "") + sec;
  }
  if (ppBtn) ppBtn.addEventListener("click", function () {
    if (srcMedia.paused) srcMedia.play(); else srcMedia.pause();
  });
  srcMedia.addEventListener("timeupdate", function () {
    if (!srcMedia.duration) return;
    if (timeline) timeline.value = Math.round((srcMedia.currentTime / srcMedia.duration) * 1000);
    if (timeLabel) timeLabel.textContent = fmt(srcMedia.currentTime) + " / " + fmt(srcMedia.duration);
  });
  if (timeline) timeline.addEventListener("input", function () {
    if (!srcMedia.duration) return;
    srcMedia.currentTime = (timeline.value / 1000) * srcMedia.duration;
  });

  // Cambi parametro → rielabora il frame corrente (debounce, no flood)
  form.addEventListener("input", function () {
    if (!hasSource) return;
    clearTimeout(paramTimer); paramTimer = setTimeout(refreshPreview, 250);
  });
  form.addEventListener("change", function (e) {
    if (!hasSource || e.target === fileInput) return;
    clearTimeout(paramTimer); paramTimer = setTimeout(refreshPreview, 120);
  });

  // --- Invio: salvataggi (fetch, senza ricaricare) + render video ------------
  function configFormData() {
    // Solo i campi di config (niente file): per salvare creazione/preset non
    // serve rispedire la sorgente. Il csrf_token viaggia coi campi hidden.
    var fd = new FormData();
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || el.type === "file" || el.type === "submit") return;
      if (el.type === "checkbox") { if (el.checked) fd.append(el.name, el.value || "y"); return; }
      fd.append(el.name, el.value);
    });
    return fd;
  }

  function postAction(fd) {
    return fetch(window.location.pathname, {
      method: "POST", headers: { "X-Requested-With": "fetch" }, body: fd,
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.error || "HTTP " + r.status);
        return d;
      });
    });
  }

  function setProgress(fraction) {
    var pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    if (progressFill) progressFill.style.width = pct + "%";
    if (progressLabel) progressLabel.textContent = pct + "%";
  }

  function pollStatus(job) {
    fetch("/video/status/" + job).then(function (r) { return r.json(); }).then(function (s) {
      if (s.state === "done") { setProgress(1); window.location.href = s.redirect; return; }
      if (s.state === "error") {
        if (progressBox) progressBox.hidden = true;
        showError("Elaborazione fallita: " + (s.error || "errore sconosciuto"));
        return;
      }
      setProgress(s.progress || 0);
      polling = setTimeout(function () { pollStatus(job); }, 1000);
    }).catch(function () { polling = setTimeout(function () { pollStatus(job); }, 2000); });
  }

  form.addEventListener("submit", function (e) {
    var action = (e.submitter && e.submitter.value) || "preview";
    e.preventDefault();
    showError("");

    if (action === "render_video") {
      if (!isVideo || !fileInput.files.length) { showError("Carica un video per l'elaborazione."); return; }
      if (progressBox) { progressBox.hidden = false; setProgress(0); }
      var fd = new FormData(form); fd.set("action", "render_video");
      postAction(fd).then(function (d) { pollStatus(d.job); })
        .catch(function (err) { if (progressBox) progressBox.hidden = true; showError(err.message); });
      return;
    }

    if (action === "save") {
      var raw = captureRawFrame();
      if (!raw) { showError("Carica una sorgente da salvare."); return; }
      if (snapshotField) snapshotField.value = raw;
      var fd2 = configFormData(); fd2.set("action", "save"); fd2.set("snapshot_raw", raw);
      postAction(fd2).then(function (d) { if (d.redirect) window.location.href = d.redirect; })
        .catch(function (err) { showError(err.message); });
      return;
    }

    if (action === "save_preset") {
      var fd3 = configFormData(); fd3.set("action", "save_preset");
      postAction(fd3).then(function (d) { if (d.redirect) window.location.href = d.redirect; })
        .catch(function (err) { showError(err.message); });
      return;
    }

    // "preview" senza JS non arriva qui (con JS l'anteprima è live): rinfresca
    refreshPreview();
  });

  window.addEventListener("beforeunload", function () {
    if (polling) clearTimeout(polling);
    if (srcUrl) URL.revokeObjectURL(srcUrl);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
})();
