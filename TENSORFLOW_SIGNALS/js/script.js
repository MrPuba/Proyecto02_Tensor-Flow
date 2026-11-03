/* Manos que hablan - Clasificador de alfabeto manual con TensorFlow.js
   MobileNet (extractor de características) + KNN (clasificador entrenable) */

const LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","LL","M","N","Ñ","O","P","Q","R","RR","S","T","U","V","W","X","Y","Z"];
const MOVING = new Set(["J","K","LL","Ñ","Q","RR","X","Z"]);
const SPOKEN = { "Ñ": "eñe", "LL": "elle", "RR": "erre" };
const BG = "FONDO"; // Clase de fondo / sin seña
const LABELS = [...LETTERS, BG];

const STORAGE_MODEL = "senas_modelo_v1";
const STORAGE_WORDS = "senas_palabras_v1";

const CONF_MIN = 0.8;      
const FRAMES_HOLD = 12;    
const K = 5;
const ROI = 0.8;           
const HISTORY = 7;         
const history = [];

function getFrame() {
    return tf.tidy(() => {
        const f = tf.browser.fromPixels(videoEl);
        const [h, w] = f.shape;
        const side = Math.floor(Math.min(h, w) * ROI);
        const y = Math.floor((h - side) / 2), x = Math.floor((w - side) / 2);
        return f.slice([y, x, 0], [side, side, 3]);
    });
}

function smooth(label, conf) {
    history.push({ label, conf });
    if (history.length > HISTORY) history.shift();
    const votes = {};
    history.forEach((h) => { (votes[h.label] = votes[h.label] || []).push(h.conf); });
    let best = null;
    Object.keys(votes).forEach((l) => { if (!best || votes[l].length > votes[best].length) best = l; });
    if (votes[best].length < Math.ceil(HISTORY / 2)) return { label: null, conf: 0 };
    return { label: best, conf: votes[best].reduce((a, b) => a + b, 0) / votes[best].length };
}

const $ = (id) => document.getElementById(id);
const videoEl = $("webcam");
const statusEl = $("status");
const wordEl = $("word");

const classifier = knnClassifier.create();
let net, webcam;
let selected = "A";
let currentLabel = null;
let streak = 0;
let committed = false;
let word = []; 
let captureTimer = null;
let savedWords = [];

const cells = {};
function buildGrid() {
    const grid = $("grid");
    LABELS.forEach((l) => {
        const c = document.createElement("div");
        c.className = "cell" + (l === BG ? " bg" : "") + (MOVING.has(l) ? " mov" : "");
        c.title = MOVING.has(l) ? l + " (con movimiento)" : (l === BG ? "Mano relajada o fuera de cuadro" : l);
        c.tabIndex = 0;
        c.setAttribute("role", "option");
        c.innerHTML = `${l === BG ? "Sin seña" : l}<small>0</small>`;
        c.onclick = () => selectLabel(l);
        c.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectLabel(l); } };
        grid.appendChild(c);
        cells[l] = c;
    });
    selectLabel("A");
}

function selectLabel(l) {
    selected = l;
    Object.entries(cells).forEach(([k, c]) => c.classList.toggle("sel", k === l));
    $("selLabel").textContent = l === BG ? "Sin seña" : l;
    $("moveHint").textContent = MOVING.has(l)
        ? "La " + l + " lleva movimiento: mantén presionado “Capturar” mientras haces todo el gesto, varias veces."
        : l === BG ? "Captura tu mano relajada, el fondo y distintas posiciones sin hacer seña."
        : "Seña estática: mantén la pose firme y varía un poco el ángulo mientras capturas.";
}

function refreshCounts() {
    const counts = classifier.getClassExampleCount();
    LABELS.forEach((l) => {
        const n = counts[l] || 0;
        cells[l].querySelector("small").textContent = n;
        cells[l].classList.toggle("ready", n >= 10);
    });
    const ready = LETTERS.filter((l) => (counts[l] || 0) >= 10).length;
    $("progressText").textContent = ready + " de " + LETTERS.length + " letras entrenadas";
    $("progressBar").style.width = (ready / LETTERS.length * 100) + "%";
}

function addExample() {
    if (!net || !webcam) { statusEl.textContent = "Espera a que el modelo y la cámara estén listos."; return; }
    tf.tidy(() => {
        const img = getFrame();
        const activation = net.infer(img, true);
        classifier.addExample(activation, selected);
    });
    refreshCounts();
}

function startCapture() {
    addExample();
    captureTimer = setInterval(addExample, 120);
}
function stopCapture() {
    if (captureTimer) { clearInterval(captureTimer); captureTimer = null; saveModel(); }
}

function serializeModel() {
    const ds = classifier.getClassifierDataset();
    const out = {};
    Object.keys(ds).forEach((k) => {
        out[k] = { data: Array.from(ds[k].dataSync()), shape: ds[k].shape };
    });
    return out;
}

function loadModel(obj) {
    classifier.clearAllClasses();
    const ds = {};
    Object.keys(obj).forEach((k) => { ds[k] = tf.tensor(obj[k].data, obj[k].shape); });
    classifier.setClassifierDataset(ds);
    refreshCounts();
}

function saveModel() {
    try {
        if (classifier.getNumClasses() === 0) { localStorage.removeItem(STORAGE_MODEL); return; }
        localStorage.setItem(STORAGE_MODEL, JSON.stringify(serializeModel()));
        $("saveInfo").textContent = "Conocimiento guardado en este navegador.";
    } catch (e) {
        $("saveInfo").textContent = "No se pudo guardar en el navegador (memoria llena). Usa “Exportar modelo”.";
    }
}

function restoreModel() {
    try {
        const raw = localStorage.getItem(STORAGE_MODEL);
        if (raw) { loadModel(JSON.parse(raw)); $("saveInfo").textContent = "Se recuperó el conocimiento guardado."; }
    } catch (e) { console.warn("No se pudo restaurar el modelo", e); }
}

function exportModel() {
    const blob = new Blob([JSON.stringify(serializeModel())], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "modelo-senas.json";
    a.click();
    URL.revokeObjectURL(a.href);
}

function importModel(file) {
    const reader = new FileReader();
    reader.onload = () => {
        try { loadModel(JSON.parse(reader.result)); saveModel(); }
        catch (e) { $("saveInfo").textContent = "El archivo no es un modelo válido."; }
    };
    reader.readAsText(file);
}

function renderWord() { wordEl.textContent = word.join(""); }

function renderSavedWords() {
    const ul = $("words");
    ul.innerHTML = "";
    savedWords.forEach((w, i) => {
        const li = document.createElement("li");
        li.textContent = w;
        const x = document.createElement("button");
        x.textContent = "×";
        x.setAttribute("aria-label", "Eliminar " + w);
        x.onclick = () => { savedWords.splice(i, 1); persistWords(); renderSavedWords(); };
        li.appendChild(x);
        ul.appendChild(li);
    });
}
function persistWords() { try { localStorage.setItem(STORAGE_WORDS, JSON.stringify(savedWords)); } catch (e) {} }
function restoreWords() {
    try { savedWords = JSON.parse(localStorage.getItem(STORAGE_WORDS) || "[]"); } catch (e) { savedWords = []; }
    renderSavedWords();
}

function speak(text) {
    if (!text.trim() || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "es-MX";
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
}

function updateBadge(label, conf) {
    const badge = $("badge");
    if (!label || label === BG) {
        $("bigLetter").textContent = "·";
        $("bigConf").textContent = label === BG ? "Sin seña" : "Esperando…";
        badge.classList.remove("hit");
    } else {
        $("bigLetter").textContent = label;
        $("bigConf").textContent = (conf * 100).toFixed(0) + "% seguro";
        badge.classList.toggle("hit", conf >= CONF_MIN);
    }
    LABELS.forEach((l) => cells[l].classList.toggle("live", l === label && conf >= CONF_MIN));
}

function handleLabel(label, conf) {
    updateBadge(label, conf);
    const valid = label && conf >= CONF_MIN;
    if (!valid || label === BG) { streak = 0; currentLabel = null; committed = false; return; }
    if (label === currentLabel) streak++; else { currentLabel = label; streak = 1; committed = false; }
    if (streak >= (MOVING.has(label) ? Math.ceil(FRAMES_HOLD / 2) : FRAMES_HOLD) && !committed) {
        word.push(label);
        committed = true;
        renderWord();
    }
}

async function loop() {
    while (true) {
        if (!videoEl.paused && !videoEl.ended) {
            const img = getFrame();
            try {
                if ($("mobi").open) {
                    const result = await net.classify(img);
                    $("console").innerHTML = "<strong>" + result[0].className + "</strong> · " + (result[0].probability * 100).toFixed(1) + "%";
                }

                // Se requieren al menos 2 clases para que KNN funcione correctamente
                if (classifier.getNumClasses() > 1) {
                    const activation = net.infer(img, true);
                    try {
                        const p = await classifier.predictClass(activation, K);
                        if (p && p.label) {
                            const s = smooth(p.label, p.confidences[p.label]);
                            handleLabel(s.label, s.conf);
                        }
                    } catch (err) {
                        console.warn("Esperando más variedad de clases...", err);
                    } finally { 
                        activation.dispose(); 
                    }
                } else {
                    updateBadge(null, 0);
                    statusEl.textContent = "Entrena al menos una letra más y la opción 'Sin seña' para comenzar.";
                }
            } catch (e) {
                console.error(e);
            } finally { 
                img.dispose(); 
            }
        }
        await tf.nextFrame();
    }
}

async function app() {
    buildGrid();
    restoreWords();
    try {
        statusEl.textContent = "Cargando modelo MobileNet…";
        net = await mobilenet.load({ version: 2, alpha: 1.0 });
        restoreModel();
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
            videoEl.srcObject = stream;
            await videoEl.play();
            webcam = true;
        } catch (e) {
            statusEl.textContent = "No se pudo acceder a la cámara. Revisa los permisos del navegador.";
            console.error(e);
            return;
        }
        statusEl.textContent = classifier.getNumClasses() > 1 ? "Listo. Haz una seña." : "Listo. Empieza entrenando una letra y 'Sin seña'.";
        loop();
    } catch (e) {
        console.error(e);
        statusEl.textContent = "Error al cargar: " + e.message;
    }
}

const cap = $("btnCapture");
cap.addEventListener("pointerdown", (e) => { e.preventDefault(); startCapture(); });
["pointerup", "pointerleave", "pointercancel"].forEach((ev) => cap.addEventListener(ev, stopCapture));
cap.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !e.repeat) { e.preventDefault(); addExample(); } });
cap.addEventListener("keyup", (e) => { if (e.key === "Enter" || e.key === " ") saveModel(); });

$("btnResetClass").onclick = () => { classifier.clearClass(selected); refreshCounts(); saveModel(); };
$("btnWipe").onclick = () => {
    if (confirm("¿Borrar todo lo que ha aprendido la aplicación?")) { classifier.clearAllClasses(); refreshCounts(); saveModel(); }
};
$("btnExport").onclick = exportModel;
$("btnImport").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => { if (e.target.files[0]) importModel(e.target.files[0]); e.target.value = ""; };

$("btnSpace").onclick = () => { word.push(" "); renderWord(); };
$("btnBack").onclick = () => { word.pop(); renderWord(); };
$("btnClear").onclick = () => { word = []; renderWord(); };
$("btnSpeak").onclick = () => speak(word.map((l) => SPOKEN[l] || l.toLowerCase()).join(" "));
$("btnSaveWord").onclick = () => {
    const w = word.join("").trim();
    if (!w) return;
    if (!savedWords.includes(w)) { savedWords.push(w); persistWords(); renderSavedWords(); }
    word = []; renderWord();
};

$("btnGuide").onclick = () => $("guide").showModal();
$("btnCloseGuide").onclick = () => $("guide").close();

function updateThemeBtn() {
    const dark = document.documentElement.dataset.theme === "dark";
    $("btnTheme").textContent = dark ? "☀ Modo día" : "☾ Modo noche";
    $("btnTheme").setAttribute("aria-label", dark ? "Cambiar a modo día" : "Cambiar a modo noche");
}
$("btnTheme").onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("senas_tema", t); } catch (e) {}
    updateThemeBtn();
};
updateThemeBtn();

app();