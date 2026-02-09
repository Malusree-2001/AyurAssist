// app.js
(() => {
  const $ = (sel) => document.querySelector(sel);

  const symptomsEl = $("#symptoms");
  const pillsEl = $("#entityPills");
  const tokenStreamEl = $("#tokenStream");
  const tokenCountEl = $("#tokenCount");
  const btnClear = $("#btnClear");

  const snomedCodeEl = $("#snomedCode");
  const snomedLabelEl = $("#snomedLabel");
  const itaCodeEl = $("#itaCode");
  const itaLabelEl = $("#itaLabel");

  const doshaNodeEl = $("#doshaNode");
  const srotasNodeEl = $("#srotasNode");
  const conditionNodeEl = $("#conditionNode");
  const pathoNoteEl = $("#pathoNote");

  const rupaListEl = $("#rupaList");
  const alertListEl = $("#alertList");

  const progIndicatorEl = $("#progIndicator");
  const progTextEl = $("#progText");

  const yearEl = $("#year");
  yearEl.textContent = String(new Date().getFullYear());

  // Keyword groups (simple rule-based NER)
  const ENTITY_RULES = [
    { key: "Breathing Difficulty", keywords: ["breathing", "breathless", "dyspnea", "shortness"], domain: "resp" },
    { key: "Wheezing", keywords: ["wheeze", "wheezing"], domain: "resp" },
    { key: "Cough", keywords: ["cough", "coughing"], domain: "resp" },
    { key: "Chest Tightness", keywords: ["tightness", "chest tight"], domain: "resp" },

    { key: "Vomiting", keywords: ["vomit", "vomiting", "throwing up", "emesis"], domain: "gi" },
    { key: "Nausea", keywords: ["nausea", "queasy"], domain: "gi" },

    { key: "Fever", keywords: ["fever", "temperature", "febrile"], domain: "gen" },
    { key: "Pain", keywords: ["pain", "ache", "aching"], domain: "gen" },
    { key: "Severe", keywords: ["severe", "intense"], domain: "severity" },
  ];

  // deterministic "confidence" (92–99) from stable hash
  function confidenceFor(label, hitCount) {
    const s = `${label}:${hitCount}`;
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return 92 + (h % 8); // 92..99
  }

  function normalizeText(t) {
    return t
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s'-]+/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function tokenize(raw) {
    // Keep punctuation-separated tokens somewhat readable
    const cleaned = raw
      .replace(/\s+/g, " ")
      .trim();
    if (!cleaned) return [];
    return cleaned.split(" ").slice(0, 18);
  }

  function findEntities(raw) {
    const text = normalizeText(raw);
    if (!text) return [];

    const entities = [];
    for (const rule of ENTITY_RULES) {
      let hits = 0;
      for (const kw of rule.keywords) {
        // word boundary-ish
        const re = new RegExp(`(^|\\s)${escapeRegExp(kw)}(\\s|$)`, "i");
        if (re.test(text)) hits++;
      }
      if (hits > 0) {
        entities.push({
          label: rule.key,
          domain: rule.domain,
          conf: confidenceFor(rule.key, hits),
        });
      }
    }

    // Sort by confidence desc, then label
    entities.sort((a, b) => (b.conf - a.conf) || a.label.localeCompare(b.label));
    return entities;
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function renderPills(entities) {
    pillsEl.innerHTML = "";
    if (entities.length === 0) {
      const empty = document.createElement("div");
      empty.className = "small subtle";
      empty.textContent = "No entities detected yet.";
      pillsEl.appendChild(empty);
      return;
    }

    for (const e of entities) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.setAttribute("role", "listitem");
      pill.setAttribute("aria-label", `${e.label} confidence ${e.conf} percent`);

      pill.innerHTML = `<strong>${e.label}</strong><span class="conf">· ${e.conf}%</span>`;
      pillsEl.appendChild(pill);
    }
  }

  function renderTokens(raw, entities) {
    tokenStreamEl.innerHTML = "";
    const toks = tokenize(raw);
    tokenCountEl.textContent = `${toks.length} tokens`;

    if (toks.length === 0) {
      const empty = document.createElement("div");
      empty.className = "small subtle";
      empty.textContent = "Start typing to see token highlights.";
      tokenStreamEl.appendChild(empty);
      return;
    }

    const hitKeywords = buildHitKeywordSet(entities);

    for (const t of toks) {
      const chip = document.createElement("span");
      chip.className = "token";
      chip.textContent = t;

      const norm = normalizeText(t);
      if (hitKeywords.has(norm)) chip.classList.add("hit");

      tokenStreamEl.appendChild(chip);
    }
  }

  function buildHitKeywordSet(entities) {
    const set = new Set();
    // for each detected entity, add its keywords normalized
    const detectedLabels = new Set(entities.map(e => e.label));
    for (const rule of ENTITY_RULES) {
      if (!detectedLabels.has(rule.key)) continue;
      for (const kw of rule.keywords) {
        set.add(normalizeText(kw));
      }
    }
    return set;
  }

  function hasDomain(entities, domain) {
    return entities.some(e => e.domain === domain);
  }

  function updateMapping(entities) {
    const isResp = hasDomain(entities, "resp");
    const isGI = hasDomain(entities, "gi");

    if (isResp) {
      // Respiratory example
      snomedCodeEl.textContent = "267036007";
      snomedLabelEl.textContent = "Respiratory distress (example)";
      itaCodeEl.textContent = "ITA-5.20.1";
      itaLabelEl.textContent = "Śvāsaḥ";

      doshaNodeEl.textContent = "Vata ↑ + Kapha ↑";
      srotasNodeEl.textContent = "Prāṇavaha Srotas";
      conditionNodeEl.textContent = "Śvāsa";

      pathoNoteEl.textContent = "Conceptual flow for respiratory presentation; validate against authoritative sources.";
      updateRupa([
        "Dyspnea / shortness of breath",
        "Wheeze, chest tightness",
        "Cough, difficulty speaking full sentences",
      ]);
      updateAlerts([
        "Severe difficulty breathing or cyanosis",
        "Chest pain, fainting, or confusion",
        "Rapid worsening despite care",
      ]);
      setPrognosis("yapya");
      return;
    }

    if (isGI) {
      // Vomiting example (avoid fabricating exact ITA code if unsure)
      snomedCodeEl.textContent = "422400008";
      snomedLabelEl.textContent = "Vomiting (example)";
      itaCodeEl.textContent = "ITA-Example";
      itaLabelEl.textContent = "Chardi (Vomiting)";

      doshaNodeEl.textContent = "Vata ↑ + Pitta ↑";
      srotasNodeEl.textContent = "Annavaha Srotas";
      conditionNodeEl.textContent = "Chardi";

      pathoNoteEl.textContent = "GI presentation example mapping; replace ITA details with verified terminology set.";
      updateRupa([
        "Nausea, retching",
        "Repeated emesis",
        "Dehydration signs (if severe)",
      ]);
      updateAlerts([
        "Blood in vomit or black/tarry stool",
        "Severe dehydration, fainting, confusion",
        "Persistent vomiting > 24 hours (or as clinically defined)",
      ]);
      setPrognosis("yapya");
      return;
    }

    // Neutral fallback
    snomedCodeEl.textContent = "—";
    snomedLabelEl.textContent = "Awaiting mapping";
    itaCodeEl.textContent = "—";
    itaLabelEl.textContent = "Awaiting mapping";

    doshaNodeEl.textContent = "Dosha pattern";
    srotasNodeEl.textContent = "Affected Srotas";
    conditionNodeEl.textContent = "Condition concept";

    pathoNoteEl.textContent = "Type symptoms to activate an example mapping (respiratory or GI).";
    updateRupa([
      "Provide a symptom narrative to detect entities.",
      "Detected entities will populate relevant sections.",
      "Mapping will appear when keywords match known demos.",
    ]);
    updateAlerts([
      "If symptoms are severe or rapidly worsening, seek urgent medical care.",
      "This system is educational and not a diagnosis.",
      "Validate with clinicians and authoritative references.",
    ]);
    setPrognosis("neutral");
  }

  function updateRupa(items) {
    rupaListEl.innerHTML = "";
    for (const it of items) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="dot dot-sage" aria-hidden="true"></span>${it}`;
      rupaListEl.appendChild(li);
    }
  }

  function updateAlerts(items) {
    alertListEl.innerHTML = "";
    for (const it of items) {
      const li = document.createElement("li");
      li.textContent = it;
      alertListEl.appendChild(li);
    }
  }

  function setPrognosis(mode) {
    // indicator position: 25% = Yapya, 75% = Asadhya
    if (mode === "yapya") {
      progIndicatorEl.style.left = "25%";
      progTextEl.textContent = "Status: Yāpya (supportable/managed) — example.";
      return;
    }
    if (mode === "asadhya") {
      progIndicatorEl.style.left = "75%";
      progTextEl.textContent = "Status: Asādhya (difficult/critical) — example.";
      return;
    }
    progIndicatorEl.style.left = "50%";
    progTextEl.textContent = "Status: Not set — enter symptoms to populate an example prognosis.";
  }

  function onInput() {
    const raw = symptomsEl.value || "";
    const entities = findEntities(raw);

    renderPills(entities);
    renderTokens(raw, entities);
    updateMapping(entities);
  }

  btnClear.addEventListener("click", () => {
    symptomsEl.value = "";
    onInput();
    symptomsEl.focus();
  });

  // Initial render with a gentle default example
  const starter = "I had wheezing and difficulty breathing yesterday.";
  symptomsEl.value = starter;
  onInput();

  symptomsEl.addEventListener("input", onInput);
})();
