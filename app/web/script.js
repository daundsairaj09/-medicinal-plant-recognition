// Clean frontend script: live upload to /predict, preview and demo fallback
"use strict";

// If serving frontend separately, set API_BASE to backend URL. You can override
// by setting window.API_BASE in the browser before this script loads.
const API_BASE = (window.API_BASE && window.API_BASE.replace(/\/$/, '')) || 'http://127.0.0.1:8000';

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseButton = document.getElementById('browseButton');
const analyzeButton = document.getElementById('analyzeButton');
const demoButton = document.getElementById('demoButton');
const quickDemoButton = document.getElementById('quickDemo');
const imagePreview = document.getElementById('imagePreview');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsContainer = document.getElementById('resultsContainer');

const confidenceValue = document.getElementById('confidenceValue');
const scientificName = document.getElementById('scientificName');
const commonName = document.getElementById('commonName');
const plantType = document.getElementById('plantType');
const plantFamily = document.getElementById('plantFamily');
const usesTags = document.getElementById('usesTags');
const primaryCompound = document.getElementById('primaryCompound');
const secondaryCompound = document.getElementById('secondaryCompound');
const safetyInfo = document.getElementById('safetyInfo');

let currentFile = null;

if (browseButton) browseButton.addEventListener('click', () => fileInput && fileInput.click());
if (fileInput) fileInput.addEventListener('change', handleFileSelect);
if (uploadArea) {
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); if (e.dataTransfer.files.length){ fileInput.files = e.dataTransfer.files; handleFileSelect(); }});
  // allow clicking the upload area to open file picker
  uploadArea.addEventListener('click', () => fileInput && fileInput.click());
}
if (analyzeButton) analyzeButton.addEventListener('click', analyzePlant);
if (demoButton) demoButton.addEventListener('click', loadDemo);
if (quickDemoButton) quickDemoButton.addEventListener('click', () => { loadDemo(); const el=document.getElementById('upload'); el && el.scrollIntoView({behavior:'smooth'}); });

// Export buttons (TXT/CSV/PDF)
const exportTxtBtn = document.getElementById('exportTxt');
const exportCsvBtn = document.getElementById('exportCsv');
const exportPdfBtn = document.getElementById('exportPdf');

if (exportTxtBtn) exportTxtBtn.addEventListener('click', () => {
  const r = getCurrentResult();
  const txt = `Plant Report\n\nScientific Name: ${r.scientificName}\nCommon Name: ${r.commonName}\nConfidence: ${r.confidence}\nType: ${r.plantType}\nFamily: ${r.plantFamily}\nPrimary Compounds: ${r.primaryCompound}\nSecondary Compounds: ${r.secondaryCompound}\nSafety: ${r.safety}\nUses: ${r.uses.join(', ')}`;
  const blob = new Blob([txt], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `${r.commonName || 'plant'}-report.txt`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
});

if (exportCsvBtn) exportCsvBtn.addEventListener('click', () => {
  const r = getCurrentResult();
  const headers = ['scientificName','commonName','confidence','plantType','plantFamily','primaryCompound','secondaryCompound','safety','uses'];
  const row = [r.scientificName, r.commonName, r.confidence, r.plantType, r.plantFamily, r.primaryCompound, r.secondaryCompound, r.safety, `"${r.uses.join('; ')}"`];
  const csv = headers.join(',') + '\n' + row.join(',');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `${r.commonName || 'plant'}-data.csv`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
});

if (exportPdfBtn) exportPdfBtn.addEventListener('click', () => {
  const r = getCurrentResult();
  const w = window.open('', '_blank');
  const html = `<html><head><title>Plant Report</title></head><body><h1>${r.commonName}</h1><p><strong>Scientific:</strong> ${r.scientificName}</p><p><strong>Confidence:</strong> ${r.confidence}</p><p><strong>Uses:</strong> ${r.uses.join(', ')}</p></body></html>`;
  w.document.write(html);
  w.document.close();
  w.print();
});

function getCurrentResult(){
  return {
    scientificName: (scientificName && scientificName.textContent) || '',
    commonName: (commonName && commonName.textContent) || '',
    confidence: (confidenceValue && confidenceValue.textContent) || '',
    plantType: (plantType && plantType.textContent) || '',
    plantFamily: (plantFamily && plantFamily.textContent) || '',
    primaryCompound: (primaryCompound && primaryCompound.textContent) || '',
    secondaryCompound: (secondaryCompound && secondaryCompound.textContent) || '',
    safety: (safetyInfo && safetyInfo.textContent) || '',
    uses: Array.from((usesTags && usesTags.querySelectorAll('.tag')) || []).map(s=>s.textContent)
  };
}

function handleFileSelect(){
  const file = fileInput && fileInput.files && fileInput.files[0];
  if (!file) return;
  if (!file.type || !file.type.startsWith('image/')) { alert('Please select an image file'); return; }
  if (file.size > 5*1024*1024){ alert('Max 5MB allowed'); return; }
  currentFile = file;
  const r = new FileReader();
  r.onload = e => {
    if (imagePreview) imagePreview.innerHTML = `<h4>Preview:</h4><img src="${e.target.result}" alt="preview" style="max-width:100%;border-radius:8px"><p class="preview-info">${file.name} (${(file.size/1024).toFixed(1)} KB)</p>`;
    if (analyzeButton) analyzeButton.disabled = false;
  };
  r.readAsDataURL(file);
}

async function analyzePlant(){
  if (!currentFile){ showNotification('No file selected — loading demo', 'info'); loadDemo(); return; }
  if (loadingIndicator) loadingIndicator.style.display = 'block';
  if (resultsContainer) resultsContainer.style.display='none';
  if (analyzeButton){ analyzeButton.disabled = true; analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...'; }

  try {
    const fd = new FormData(); fd.append('file', currentFile);
    const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Server: '+res.status);
    const data = await res.json();
    const info = data.info || {};
    const result = {
      confidence: data.confidence || 0,
      scientificName: info.scientific_name || data.plant_id || 'Unknown',
      commonName: info.common_name || data.plant_id || 'Unknown',
      plantType: info.plant_type || 'Unknown',
      plantFamily: info.family || 'Unknown',
      uses: info.uses || info.medicinal_uses || [],
      primaryCompound: info.primary_compound || '-',
      secondaryCompound: info.secondary_compounds || '-',
      safety: info.safety || '-'
    };
    updateResults(result);
    showNotification(`Detected ${result.commonName} — ${(result.confidence*100).toFixed(1)}%`, 'success');
  } catch (err) {
    console.error(err);
    showNotification('Live analysis failed — showing demo', 'error');
    loadDemo();
  } finally {
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    if (analyzeButton){ analyzeButton.disabled = false; analyzeButton.innerHTML = '<i class="fas fa-search"></i> Analyze Plant'; }
    if (resultsContainer) resultsContainer.style.display = 'block';
  }
}

function loadDemo(){
  const plant = {
    scientificName: 'Ocimum tenuiflorum', commonName:'Tulsi (Holy Basil)', plantType:'Medicinal Herb', plantFamily:'Lamiaceae', confidence:0.72,
    uses:['Adaptogen','Respiratory Relief','Anti-inflammatory'], primaryCompound:'Eugenol', secondaryCompound:'Ursolic Acid', safety:'Consult doctor if on blood thinners.'
  };
  if (imagePreview) imagePreview.innerHTML = `<h4>Demo Image (Tulsi):</h4><img src="https://images.unsplash.com/photo-1598257003911-5bdcbce4bc56?auto=format&fit=crop&w=500&q=80" alt="demo" style="max-width:100%;border-radius:8px"><p class="preview-info">Demo image loaded</p>`;
  updateResults(plant);
  if (resultsContainer) resultsContainer.style.display='block';
}

function updateResults(p){
  if (confidenceValue) confidenceValue.textContent = (p.confidence||0).toFixed(2);
  if (scientificName) scientificName.textContent = p.scientificName || '-';
  if (commonName) commonName.textContent = p.commonName || '-';
  if (plantType) plantType.textContent = p.plantType || '-';
  if (plantFamily) plantFamily.textContent = p.plantFamily || '-';
  if (primaryCompound) primaryCompound.textContent = p.primaryCompound || '-';
  if (secondaryCompound) secondaryCompound.textContent = p.secondaryCompound || '-';
  if (safetyInfo) safetyInfo.textContent = p.safety || '-';
  if (usesTags) { usesTags.innerHTML = ''; (p.uses||[]).forEach(u=>{ const s=document.createElement('span'); s.className='tag'; s.textContent=u; usesTags.appendChild(s); }); }
}

function showNotification(msg, type='info'){
  const ex = document.querySelector('.notification'); if (ex) ex.remove();
  const n = document.createElement('div'); n.className = 'notification'; n.textContent = msg; document.body.appendChild(n);
  setTimeout(()=> n.remove(), 3500);
}

// Initialize demo on load
window.addEventListener('DOMContentLoaded', () => { loadDemo(); setTimeout(()=> showNotification('Welcome to Leaf Detector — upload an image or try demo', 'info'), 700); });
