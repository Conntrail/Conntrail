/**
 * Cloudflare Pages Function — POST /api/contact
 *
 * Receives the contact form, validates it, and forwards it to your email via
 * Resend (or a generic webhook). Secrets live in Pages environment variables,
 * never in the repo.
 *
 * Required env (one of):
 *   RESEND_API_KEY      + CONTACT_TO_EMAIL (default hello@ibzie.dev)
 *                         + CONTACT_FROM_EMAIL (a verified Resend sender)
 *   CONTACT_WEBHOOK_URL (any endpoint that accepts a JSON POST)
 *
 * Optional:
 *   TURNSTILE_SECRET    Cloudflare Turnstile secret to verify the token.
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

function json(body, status) {
  return new Response(JSON.stringify(body), { status: status || 200, headers: JSON_HEADERS });
}

function isEmail(value) {
  return typeof value === "string" && /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(value);
}

export async function onRequestPost(context) {
  const { request, env } = context;

  let data;
  try {
    data = await request.json();
  } catch (err) {
    return json({ ok: false, error: "Invalid JSON body." }, 400);
  }

  const name = (data.name || "").toString().trim().slice(0, 200);
  const email = (data.email || "").toString().trim().slice(0, 320);
  const company = (data.company || "").toString().trim().slice(0, 200);
  const message = (data.message || "").toString().trim().slice(0, 5000);
  const honeypot = (data.website || "").toString().trim();

  // Bots fill hidden fields. Pretend success so they don't retry.
  if (honeypot) return json({ ok: true });

  if (!name || !message || !isEmail(email)) {
    return json({ ok: false, error: "Name, a valid email and a message are required." }, 400);
  }

  if (env.TURNSTILE_SECRET) {
    const token = (data["cf-turnstile-response"] || "").toString();
    if (!token) return json({ ok: false, error: "Missing verification token." }, 400);
    const verify = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ secret: env.TURNSTILE_SECRET, response: token, remoteip: request.headers.get("CF-Connecting-IP") || "" })
    });
    const outcome = await verify.json();
    if (!outcome.success) return json({ ok: false, error: "Verification failed." }, 400);
  }

  const subject = "Conntrail enquiry — " + name + (company ? " (" + company + ")" : "");
  const text = [
    "Name: " + name,
    "Email: " + email,
    "Company: " + company || "—",
    "",
    message
  ].join("\n");

  try {
    if (env.RESEND_API_KEY) {
      const res = await fetch("https://api.resend.com/emails", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + env.RESEND_API_KEY,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          from: env.CONTACT_FROM_EMAIL || "Conntrail <onboarding@resend.dev>",
          to: [env.CONTACT_TO_EMAIL || "hello@ibzie.dev"],
          reply_to: email,
          subject: subject,
          text: text
        })
      });
      if (!res.ok) {
        const detail = await res.text();
        return json({ ok: false, error: "Email provider error." }, 502, detail);
      }
      return json({ ok: true });
    }

    if (env.CONTACT_WEBHOOK_URL) {
      const res = await fetch(env.CONTACT_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, company, message, subject })
      });
      if (!res.ok) return json({ ok: false, error: "Webhook error." }, 502);
      return json({ ok: true });
    }
  } catch (err) {
    return json({ ok: false, error: "Delivery failed." }, 502);
  }

  // Not configured: tell the client so it can fall back to the mailto link.
  return json({ ok: false, error: "Contact form is not configured." }, 503);
}

export async function onRequestGet() {
  return json({ ok: false, error: "Method not allowed." }, 405);
}
