// Mark the current nav link as active
document.querySelectorAll('.nav-link').forEach(link => {
  if (link.getAttribute('href') === window.location.pathname) {
    link.classList.add('active');
  }
});

// Mobile nav toggle
const navToggle = document.getElementById('nav-toggle');
const mobileNav = document.getElementById('mobile-nav');
const iconOpen  = document.getElementById('icon-open');
const iconClose = document.getElementById('icon-close');

if (navToggle) {
  navToggle.addEventListener('click', () => {
    mobileNav.classList.toggle('hidden');
    iconOpen.classList.toggle('hidden');
    iconClose.classList.toggle('hidden');
  });
}
