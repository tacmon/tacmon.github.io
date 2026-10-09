document.querySelectorAll('[data-tex]').forEach((element) => {
  if (!window.katex) return;
  katex.render(element.dataset.tex, element, {
    displayMode: element.classList.contains('display'),
    throwOnError: false,
    strict: 'warn',
    trust: false,
  });
});
