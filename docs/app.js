const input = document.getElementById("symptomInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const entityOverlay = document.getElementById("entityOverlay");
const mappingCard = document.getElementById("mappingCard");

function runEntityDetection(text) {
  entityOverlay.innerHTML = "";

  // Demo entities
  const entities = [
    { name: "Cough", conf: 93 },
    { name: "Dyspnea", conf: 89 }
  ];

  entities.forEach(e => {
    const pill = document.createElement("div");
    pill.className = "entity-pill";
    pill.textContent = `${e.name} · ${e.conf}%`;
    entityOverlay.appendChild(pill);
  });

  mappingCard.classList.remove("hidden");
}

analyzeBtn.addEventListener("click", () => {
  const text = input.value.trim();
  if (!text) return;
  runEntityDetection(text);
});

// Enter = Analyze, Shift+Enter = newline
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    analyzeBtn.click();
  }
});
