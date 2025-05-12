document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const resultsDiv = document.querySelector('.res');

    form.addEventListener('submit', function() {
        if (resultsDiv) {
            resultsDiv.style.display = 'grid';
            setTimeout(function() {
                resultsDiv.style.display = 'none';
            }, 5000);
        }
    });
});