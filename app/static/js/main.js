// BlobTrack Studio — JS lato client.

document.addEventListener("DOMContentLoaded", function () {
  setupCookieBanner();
  setupRangeOutputs();
  setupFileName();
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

// Mostra il nome del file selezionato sul pulsante di upload (Studio e Video).
function setupFileName() {
  var input = document.querySelector('#studio-form input[type="file"]');
  var label = document.querySelector(".fname");
  if (!input || !label) return;
  var fallback = label.textContent;
  input.addEventListener("change", function () {
    label.textContent = input.files && input.files.length ? input.files[0].name : fallback;
  });
}
