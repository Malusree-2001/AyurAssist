document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    const symptomInput = document.getElementById('symptomInput');
    const entityOverlay = document.getElementById('entityOverlay');
    const mappingCard = document.getElementById('mappingCard');

    // Simulated NER (Named Entity Recognition) Mapping
    const clinicalMap = {
        'breathing': 'Breathing: 98% (Sign)',
        'wheezing': 'Wheezing: 96% (Sign)',
        'chest': 'Chest Tightness: 92% (Symptom)',
        'cough': 'Cough: 85% (Symptom)'
    };

    symptomInput.addEventListener('input', (e) => {
        const val = e.target.value.toLowerCase();
        entityOverlay.innerHTML = '';
        let foundAny = false;

        for (const [key, label] of Object.entries(clinicalMap)) {
            if (val.includes(key)) {
                const pill = document.createElement('span');
                pill.className = 'pill-detect';
                pill.innerText = label;
                entityOverlay.appendChild(pill);
                foundAny = true;
            }
        }

        // Show the bridge card if relevant symptoms are found
        if (foundAny) {
            mappingCard.classList.remove('hidden');
        } else {
            mappingCard.classList.add('hidden');
        }
    });
});
