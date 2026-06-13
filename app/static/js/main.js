// BlobTrack Studio — JS lato client.

document.addEventListener("DOMContentLoaded", function () {
  setupCookieBanner();
  setupRangeOutputs();
  setupFileName();
  setupConditionalFields();
});

// Banner consenso cookie (GDPR): invio via fetch + nascondi senza ricaricare.
// Funziona anche senza JS grazie al submit classico del form (fallback server-side).
function setupCookieBanner() {
  var banner = document.getElementById("cookie-banner");
  var form = document.getElementById("cookie-form");
  if (!banner || !form) return;

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var action = (document.activeElement && document.activeElement.value) || "accept";
    var body = new URLSearchParams();
    body.append("action", action);
    fetch(form.action, { method: "POST", headers: { "X-Requested-With": "fetch" }, body: body })
      .then(function () { banner.style.display = "none"; })
      .catch(function () { form.submit(); });
  });
}

// Mostra il valore corrente accanto a ogni slider.
function setupRangeOutputs() {
  document.querySelectorAll(".js-range").forEach(function (range) {
    var field = range.closest(".field");
    var out = field ? field.querySelector(".val") : null;
    if (!out) return;
    var sync = function () { out.textContent = range.value; };
    range.addEventListener("input", sync);
    sync();
  });
}

// Mostra/nasconde gruppi di controlli in base al valore di un altro campo.
// Dichiarativo, via data-attribute sul contenitore (riusabile su Studio/Video/Live):
//   data-show-when="campo:val1|val2"   -> visibile se campo ha uno di quei valori
//   data-show-when-checked="campo"     -> visibile se la checkbox è spuntata
//   data-show-when-file="campo"        -> visibile se è stato scelto un file
function setupConditionalFields() {
  var form = document.getElementById("studio-form") || document.getElementById("live-form");
  if (!form) return;
  var conds = form.querySelectorAll("[data-show-when], [data-show-when-checked], [data-show-when-file]");
  if (!conds.length) return;

  function field(name) { return form.querySelector('[name="' + name + '"]'); }

  function refresh() {
    conds.forEach(function (el) {
      var show = true;
      var sw = el.getAttribute("data-show-when");
      if (sw) {
        var parts = sw.split(":");
        var ctrl = field(parts[0]);
        show = !!ctrl && (parts[1] || "").split("|").indexOf(ctrl.value) !== -1;
      }
      var ck = el.getAttribute("data-show-when-checked");
      if (ck) { var c = field(ck); show = show && !!(c && c.checked); }
      var sf = el.getAttribute("data-show-when-file");
      if (sf) { var f = field(sf); show = show && !!(f && f.files && f.files.length); }
      el.hidden = !show;
    });
  }

  form.addEventListener("change", refresh);
  refresh();
}

// Mostra il nome del file selezionato sul relativo pulsante di upload.
// Gestisce più input file (es. Video + traccia Audio) abbinandoli alla label
// tramite l'attributo `for`.
function setupFileName() {
  document.querySelectorAll('#studio-form input[type="file"]').forEach(function (input) {
    var label = document.querySelector('label[for="' + input.id + '"] .fname');
    if (!label) return;
    var fallback = label.textContent;
    input.addEventListener("change", function () {
      label.textContent = input.files && input.files.length ? input.files[0].name : fallback;
    });
  });
}
