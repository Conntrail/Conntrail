/*
 * Runtime configuration for the live demo page.
 *
 * CONNTRAIL_DEMO_API: base URL of the demo service (demo/service, on Render).
 *   Leave empty to use same-origin, or set it to your deployed service URL.
 * CONNTRAIL_TURNSTILE_SITEKEY: optional Cloudflare Turnstile site key. When
 *   set, the live form renders a bot-check widget; the server verifies it.
 */
window.CONNTRAIL_DEMO_API = "https://conntrail-demo.onrender.com";
window.CONNTRAIL_TURNSTILE_SITEKEY = "";
