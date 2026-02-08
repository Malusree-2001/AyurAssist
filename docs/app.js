const API_BASE = 'https://sunbiz--ayurparam-service-fastapi-app.modal.run';

// Wake up GPU container while the user types (fire-and-forget)
fetch(API_BASE + '/warmup').catch(() => {});

const input = document.getElementById('symptomInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');

// Example buttons
document.getElementById('examples').addEventListener('click', (e) => {
  if (e.target.classList.contains('example-btn')) {
    input.value = e.target.dataset.value;
    analyzeBtn.disabled = false;
  }
});

// Enable/disable button based on input
input.addEventListener('input', () => {
  analyzeBtn.disabled = !input.value.trim();
});

// Enter key submits
input.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && input.value.trim()) {
    analyze();
  }
});

// Click submits
analyzeBtn.addEventListener('click', analyze);

async function analyze() {
  const text = input.value.trim();
  if (!text) return;

  // Set loading state
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  loadingEl.classList.remove('hidden');
  errorEl.classList.add('hidden');
  resultsEl.classList.add('hidden');

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch(API_BASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) throw new Error('HTTP ' + response.status);

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    const msg = err.name === 'AbortError'
      ? 'Request timeout. AI is processing, please try again.'
      : 'Cannot connect to backend. Check Modal deployment.';
    errorEl.textContent = '\u26A0\uFE0F ' + msg;
    errorEl.classList.remove('hidden');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
    loadingEl.classList.add('hidden');
  }
}

function esc(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderResults(data) {
  const treatment = data.results && data.results[0] && data.results[0].treatment_info;
  if (!treatment) {
    errorEl.textContent = '\u26A0\uFE0F No treatment information found for this symptom.';
    errorEl.classList.remove('hidden');
    return;
  }

  let html = '';

  // Clinical Entities & Medical Codes
  html += '<div class="info-grid">';

  // Clinical Entities
  html += '<div class="clinical-entities-card"><h3>CLINICAL ENTITIES</h3><div class="entity-tags">';
  if (data.clinical_entities && data.clinical_entities.length) {
    data.clinical_entities.forEach((ent) => {
      html += '<span class="entity-tag">' + esc(ent.word) + '</span>';
    });
  }
  html += '</div></div>';

  // Medical Codes
  const umls = data.umls_cui || 'N/A';
  const snomed = data.snomed_code || (data.results[0] && data.results[0].snomed_code) || 'N/A';
  html += '<div class="medical-codes-card"><h3>MEDICAL CODES</h3>';
  html += '<div class="code-line"><strong>UMLS:</strong> ' + esc(umls) + '</div>';
  html += '<div class="code-line"><strong>SNOMED:</strong> ' + esc(snomed) + '</div>';
  html += '</div>';

  html += '</div>';

  // Condition Banner
  html += '<div class="condition-banner">';
  html += '<h2>' + esc(treatment.condition_name || '') + '</h2>';
  html += '<p>' + esc(treatment.sanskrit_name || '') + '</p>';
  html += '</div>';

  // Clinical Overview
  html += '<div class="card"><h3>\uD83D\uDCCB Clinical Overview</h3>';
  html += '<p>' + esc(treatment.brief_description || '') + '</p>';
  html += '<div class="overview-meta">';
  html += '<div class="meta-box"><strong>Dosha:</strong> ' + esc(treatment.dosha_involvement || '') + '</div>';
  html += '<div class="meta-box"><strong>Prognosis:</strong> ' + esc(treatment.prognosis || '') + '</div>';
  html += '</div></div>';

  // Nidana (Causes)
  if (treatment.nidana_causes && treatment.nidana_causes.length) {
    html += '<div class="card"><h3>\uD83D\uDD0D Nidana (Causes)</h3><ul>';
    treatment.nidana_causes.forEach((c) => {
      html += '<li>' + esc(c) + '</li>';
    });
    html += '</ul></div>';
  }

  // Rupa (Symptoms)
  if (treatment.rupa_symptoms && treatment.rupa_symptoms.length) {
    html += '<div class="card"><h3>\uD83E\uDE7A Rupa (Symptoms)</h3><div class="symptom-tags">';
    treatment.rupa_symptoms.forEach((s) => {
      html += '<span class="symptom-tag">' + esc(s) + '</span>';
    });
    html += '</div></div>';
  }

  // Ottamooli (Single Remedies)
  if (treatment.ottamooli_single_remedies && treatment.ottamooli_single_remedies.length) {
    html += '<div class="card"><h3>\uD83C\uDF3F Ottamooli (Single Remedies)</h3>';
    treatment.ottamooli_single_remedies.forEach((r) => {
      html += '<div class="remedy-item">';
      html += '<div class="name">' + esc(r.medicine_name || '') + '</div>';
      html += '<div class="sanskrit">' + esc(r.sanskrit_name || '') + '</div>';
      html += '<div class="remedy-details">';
      html += '<div><strong>Part:</strong> ' + esc(r.part_used || '') + '</div>';
      html += '<div><strong>Dosage:</strong> ' + esc(r.dosage || '') + '</div>';
      html += '<div><strong>Preparation:</strong> ' + esc(r.preparation || '') + '</div>';
      html += '<div><strong>Timing:</strong> ' + esc(r.timing || '') + '</div>';
      html += '<div class="full-width"><strong>Duration:</strong> ' + esc(r.duration || '') + '</div>';
      html += '</div></div>';
    });
    html += '</div>';
  }

  // Classical Formulations
  if (treatment.classical_formulations && treatment.classical_formulations.length) {
    html += '<div class="card"><h3>\uD83D\uDCDC Classical Formulations</h3>';
    treatment.classical_formulations.forEach((f) => {
      html += '<div class="formulation-item">';
      html += '<div class="name">' + esc(f.name || '') + '</div>';
      html += '<div class="english">' + esc(f.english_name || '') + '</div>';
      html += '<div class="details"><strong>Form:</strong> ' + esc(f.form || '') + ' | <strong>Dosage:</strong> ' + esc(f.dosage || '') + '</div>';
      if (f.reference_text) {
        html += '<div class="reference">\uD83D\uDCDA ' + esc(f.reference_text) + '</div>';
      }
      html += '</div>';
    });
    html += '</div>';
  }

  // Pathya (Dietary Advice)
  if (treatment.pathya_dietary_advice) {
    const diet = treatment.pathya_dietary_advice;
    html += '<div class="card"><h3>\uD83C\uDF7D\uFE0F Pathya (Dietary Advice)</h3>';
    html += '<div class="diet-grid">';

    html += '<div class="favor"><h4>\u2713 Foods to Favor</h4><ul>';
    if (diet.foods_to_favor) {
      diet.foods_to_favor.forEach((f) => { html += '<li>' + esc(f) + '</li>'; });
    }
    html += '</ul></div>';

    html += '<div class="avoid"><h4>\u2717 Foods to Avoid</h4><ul>';
    if (diet.foods_to_avoid) {
      diet.foods_to_avoid.forEach((f) => { html += '<li>' + esc(f) + '</li>'; });
    }
    html += '</ul></div>';

    html += '</div>';

    if (diet.specific_dietary_rules) {
      html += '<div class="diet-note">\uD83D\uDCA1 ' + esc(diet.specific_dietary_rules) + '</div>';
    }
    html += '</div>';
  }

  // Lifestyle & Yoga (side by side)
  const hasLifestyle = treatment.vihara_lifestyle && treatment.vihara_lifestyle.length;
  const hasYoga = treatment.yoga_exercises && treatment.yoga_exercises.length;

  if (hasLifestyle || hasYoga) {
    html += '<div class="two-col-grid">';

    if (hasLifestyle) {
      html += '<div class="card"><h3>\uD83C\uDFC3 Vihara (Lifestyle)</h3><ul style="font-size:0.875rem;line-height:1.6">';
      treatment.vihara_lifestyle.forEach((v) => { html += '<li>' + esc(v) + '</li>'; });
      html += '</ul></div>';
    }

    if (hasYoga) {
      html += '<div class="card"><h3>\uD83E\uDDD8 Yoga Exercises</h3><div class="yoga-tags">';
      treatment.yoga_exercises.forEach((y) => {
        html += '<span class="yoga-tag">' + esc(y) + '</span>';
      });
      html += '</div></div>';
    }

    html += '</div>';
  }

  // Warning Signs
  if (treatment.warning_signs && treatment.warning_signs.length) {
    html += '<div class="card"><h3>\u26A0\uFE0F Warning Signs</h3>';
    treatment.warning_signs.forEach((w) => {
      html += '<div class="warning-item">! ' + esc(w) + '</div>';
    });
    html += '</div>';
  }

  // Disclaimer
  if (treatment.disclaimer) {
    html += '<div class="disclaimer">\u2695\uFE0F ' + esc(treatment.disclaimer) + '</div>';
  }

  resultsEl.innerHTML = html;
  resultsEl.classList.remove('hidden');
}