// BlobTrack Studio — JS lato client.

document.addEventListener("DOMContentLoaded", function () {
  setupCookieBanner();
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

    fetch(form.action, {
      method: "POST",
      headers: { "X-Requested-With": "fetch" },
      body: body,
    })
      .then(function () {
        banner.style.display = "none";
      })
      .catch(function () {
        // Se il fetch fallisce, lasciamo che il form venga inviato normalmente
        form.submit();
      });
  });
}
