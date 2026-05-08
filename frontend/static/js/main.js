// ============================================
// KisanAI - Main Page JavaScript
// ============================================

// ── Navbar scroll effect ─────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 30);
});

// ── Mobile nav toggle ────────────────────────
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');
navToggle.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});
document.addEventListener('click', (e) => {
  if (!navbar.contains(e.target)) navLinks.classList.remove('open');
});

// ── Scroll animations ────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => entry.target.classList.add('visible'), i * 80);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));

// ── Animated counters ────────────────────────
function animateCounter(el, target, suffix = '') {
  const duration = 2200;
  const startTime = performance.now();
  const isLarge = target >= 1000;

  function format(n) {
    if (n >= 100000) return Math.floor(n / 1000) + 'K+';
    if (n >= 1000) return Math.floor(n / 1000) + 'K+';
    return Math.floor(n).toString();
  }

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.floor(eased * target);
    el.textContent = format(current) + suffix;
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = format(target) + suffix;
  }
  requestAnimationFrame(update);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      const target = parseInt(el.dataset.target);
      animateCounter(el, target);
      counterObserver.unobserve(el);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.impact-number[data-target]').forEach(el => counterObserver.observe(el));

// Hero stat counters
const heroCounters = [
  { id: 'cnt-farmers',  target: 50000 },
  { id: 'cnt-queries',  target: 200000 },
];
setTimeout(() => {
  heroCounters.forEach(({ id, target }) => {
    const el = document.getElementById(id);
    if (el) animateCounter(el, target);
  });
}, 600);

// ── Smooth scroll for nav links ───────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const href = a.getAttribute('href');
    if (href === '#') return;
    const target = document.querySelector(href);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      navLinks.classList.remove('open');
    }
  });
});
