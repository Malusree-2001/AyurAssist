document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    const symptomInput = document.getElementById('symptomInput');
    const entityOverlay = document.getElementById('entityOverlay');
    const mappingCard = document.getElementById('mappingCard');

    const keywords = {
        'breathing': 'Breathing: 98%',
        'wheezing': 'Wheezing: 96%',
        'breath': 'Dyspnoea: 80%'
    };

    symptomInput.addEventListener('input', (e) => {
        const text = e.target.value.toLowerCase();
        entityOverlay.innerHTML = '';
        let found = false;

        for (const [key, value] of Object.entries(keywords)) {
            if (text.includes(key)) {
                const pill = document.createElement('span');
                pill.className = 'pill';
                pill.innerText = value;
                entityOverlay.appendChild(pill);
                found = true;
            }
        }

        // Show mapping card if keywords detected
        if (found) {
            mappingCard.classList.remove('hidden');
        } else {
            mappingCard.classList.add('hidden');
        }
    });
});
