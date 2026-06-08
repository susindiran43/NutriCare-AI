// ==========================================================================
// NutriCare AI - Interactive Frontend Controller
// ==========================================================================

// Global Application State
let allSymptoms = []; // Array of symptom strings from the backend
let selectedSymptoms = new Set(); // Set of currently selected symptom strings
let catalogData = []; // Full list of nutritional knowledge entries

// Document Lifecycle Initializer
document.addEventListener("DOMContentLoaded", () => {
    fetchSymptomsList();
    
    // Close dropdown search if clicked outside
    document.addEventListener("click", (e) => {
        const dropdown = document.getElementById("symptoms-dropdown");
        const searchInput = document.getElementById("symptom-search");
        if (dropdown && !dropdown.contains(e.target) && e.target !== searchInput) {
            dropdown.style.display = "none";
        }
    });
});

// Format computer name (e.g. skin_rash) to human display (e.g. Skin Rash)
function formatSymptomDisplay(sym) {
    if (!sym) return "";
    return sym
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// ==========================================================================
// API Operations
// ==========================================================================

// Fetch unique symptoms list from backend
async function fetchSymptomsList() {
    try {
        const response = await fetch('/api/symptoms');
        if (!response.ok) throw new Error("Failed to load symptoms catalog.");
        
        const data = await response.json();
        allSymptoms = data.symptoms || [];
        console.log(`Loaded ${allSymptoms.length} symptoms successfully.`);
    } catch (err) {
        console.error("Error fetching symptoms:", err);
        showErrorToast("Could not load the symptoms catalog. Please refresh the page.");
    }
}

// Fetch entire dietary knowledge base catalog
async function fetchCatalogData() {
    if (catalogData.length > 0) return; // Prevent double load
    
    const gridContainer = document.getElementById("catalog-grid-container");
    gridContainer.innerHTML = `
        <div class="search-results-loading" style="grid-column: 1/-1;">
            <div class="spinner"></div>
            <p>Loading database directory...</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/catalog');
        if (!response.ok) throw new Error("Catalog fetch failed.");
        
        const data = await response.json();
        catalogData = data.catalog || [];
        renderCatalogGrid(catalogData);
    } catch (err) {
        console.error(err);
        gridContainer.innerHTML = `
            <div class="search-empty-state" style="grid-column: 1/-1;">
                <h3>Failed to Load Directory</h3>
                <p>Ensure the Flask backend server is running correctly.</p>
            </div>
        `;
    }
}

// Run Machine Learning inference classification & dietary retrieval
async function runPrediction() {
    if (selectedSymptoms.size === 0) {
        alert("Please select at least one symptom to run the analysis.");
        return;
    }
    
    const placeholder = document.getElementById("results-placeholder");
    const loading = document.getElementById("results-loading");
    const content = document.getElementById("results-content");
    
    // UI Transitions
    placeholder.style.display = "none";
    content.style.display = "none";
    loading.style.display = "flex";
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: Array.from(selectedSymptoms) })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Prediction request failed.");
        }
        
        const result = await response.json();
        renderPredictionResults(result);
        
    } catch (err) {
        console.error(err);
        loading.style.display = "none";
        placeholder.style.display = "flex";
        alert(err.message || "An error occurred during symptom classification.");
    }
}

// Run FAISS Semantic Vector Space Query Search
async function runSemanticSearch() {
    const input = document.getElementById("semantic-search-input");
    const query = input.value.trim();
    
    if (!query) {
        alert("Please type a search query first.");
        return;
    }
    
    const container = document.getElementById("search-results-container");
    const loading = document.getElementById("search-loading");
    
    container.innerHTML = "";
    loading.style.display = "flex";
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        
        loading.style.display = "none";
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Search failed.");
        }
        
        const data = await response.json();
        renderSearchResults(data.results || []);
        
    } catch (err) {
        console.error(err);
        loading.style.display = "none";
        container.innerHTML = `
            <div class="search-empty-state">
                <h3>Search Failed</h3>
                <p>${err.message || "Could not retrieve search result from the server."}</p>
            </div>
        `;
    }
}

// ==========================================================================
// Rendering Utilities
// ==========================================================================

// Display the ML classification and nutritional recommendations on the UI
function renderPredictionResults(data) {
    document.getElementById("results-loading").style.display = "none";
    
    // Fill text properties
    document.getElementById("result-disease-name").innerText = data.disease;
    
    // Confidence Percentage
    const confidencePct = Math.round(data.confidence * 100);
    document.getElementById("result-confidence").innerText = `${confidencePct}%`;
    document.getElementById("result-confidence-bar").style.width = `${confidencePct}%`;
    
    // Description
    document.getElementById("result-description").innerText = data.description;
    
    // Precautions Check List
    const precautionsList = document.getElementById("result-precautions-list");
    precautionsList.innerHTML = "";
    data.precautions.forEach(prec => {
        const li = document.createElement("li");
        li.innerText = prec;
        precautionsList.appendChild(li);
    });
    
    // Diet Info
    document.getElementById("nutrition-disease-title").innerText = data.nutrition.disease;
    document.getElementById("result-recommended-food").innerText = data.nutrition.recommended;
    document.getElementById("result-avoid-food").innerText = data.nutrition.avoid;
    document.getElementById("result-diet-reason").innerText = data.nutrition.reason;
    
    // Display FAISS tag match method
    const matchBadge = document.getElementById("diet-match-badge");
    if (data.nutrition.match_method === 'exact') {
        matchBadge.innerText = "Exact Match Lookup";
        matchBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
        matchBadge.style.color = "var(--success)";
    } else {
        matchBadge.innerText = "FAISS Semantic Match";
        matchBadge.style.borderColor = "rgba(0, 242, 195, 0.3)";
        matchBadge.style.color = "var(--primary)";
    }
    
    // Show details
    document.getElementById("results-content").style.display = "flex";
}

// Render search results from semantic FAISS querying
function renderSearchResults(results) {
    const container = document.getElementById("search-results-container");
    container.innerHTML = "";
    
    if (results.length === 0) {
        container.innerHTML = `
            <div class="search-empty-state">
                <h3>No Matching Recommendations</h3>
                <p>We couldn't find any dietary matches for your search. Try expressing your query differently.</p>
            </div>
        `;
        return;
    }
    
    results.forEach(res => {
        // Map Distance Score to a friendly relative match accuracy label
        const confidenceScore = Math.max(0, Math.min(100, Math.round((2.0 - res.distance) * 50)));
        
        const card = document.createElement("div");
        card.className = "search-result-card";
        card.innerHTML = `
            <div class="search-result-header">
                <h3>${res.disease}</h3>
                <span class="similarity-badge">Match Score: ${confidenceScore}%</span>
            </div>
            <div class="card-body">
                <div class="nutrition-split">
                    <div class="diet-box recommend">
                        <h5><span class="diet-icon">✔</span> Recommended Foods</h5>
                        <p>${res.recommended}</p>
                    </div>
                    <div class="diet-box avoid">
                        <h5><span class="diet-icon">✖</span> Foods to Avoid</h5>
                        <p>${res.avoid}</p>
                    </div>
                </div>
                <div class="diet-reason">
                    <h6>Clinical Rationale</h6>
                    <p>${res.reason}</p>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

// Render the grid list on catalog tab
function renderCatalogGrid(items) {
    const gridContainer = document.getElementById("catalog-grid-container");
    gridContainer.innerHTML = "";
    
    if (items.length === 0) {
        gridContainer.innerHTML = `
            <div class="search-empty-state" style="grid-column: 1/-1;">
                <h3>No Matching Catalog Items</h3>
                <p>Try reducing or resetting the search filter term.</p>
            </div>
        `;
        return;
    }
    
    items.forEach(item => {
        const card = document.createElement("div");
        card.className = "catalog-card";
        card.innerHTML = `
            <h4>${item.disease}</h4>
            <div class="catalog-details">
                <div class="catalog-info-row rec">
                    <span class="label">✔ Recommend</span>
                    <span class="val">${item.recommended}</span>
                </div>
                <div class="catalog-info-row avd">
                    <span class="label">✖ Avoid</span>
                    <span class="val">${item.avoid}</span>
                </div>
                <div class="catalog-info-row rsn">
                    <span class="label">Rationale</span>
                    <span class="val">${item.reason}</span>
                </div>
            </div>
        `;
        gridContainer.appendChild(card);
    });
}

// ==========================================================================
// Symptom List Interactive Search and Pill Generation
// ==========================================================================

// Filter symptom choices dynamically as user types
function filterSymptoms() {
    const input = document.getElementById("symptom-search");
    const query = input.value.trim().toLowerCase();
    const dropdown = document.getElementById("symptoms-dropdown");
    const clearBtn = document.getElementById("clear-search");
    
    if (!query) {
        dropdown.style.display = "none";
        clearBtn.style.display = "none";
        return;
    }
    
    clearBtn.style.display = "block";
    
    // Filter out symptoms that are already selected, and match the text query
    const filtered = allSymptoms.filter(sym => 
        !selectedSymptoms.has(sym) && 
        sym.replace(/_/g, ' ').toLowerCase().includes(query)
    );
    
    if (filtered.length === 0) {
        dropdown.innerHTML = `<div class="dropdown-item" style="color: var(--text-muted); cursor: default;">No matching symptoms found</div>`;
    } else {
        dropdown.innerHTML = filtered
            .slice(0, 8) // Show top 8 items max
            .map(sym => `
                <div class="dropdown-item" onclick="selectSymptom('${sym}')">
                    ${formatSymptomDisplay(sym)}
                </div>
            `).join('');
    }
    
    dropdown.style.display = "block";
}

// Select a symptom pill and append it to tags list
function selectSymptom(sym) {
    selectedSymptoms.add(sym);
    
    // Clear search box
    const searchInput = document.getElementById("symptom-search");
    searchInput.value = "";
    document.getElementById("symptoms-dropdown").style.display = "none";
    document.getElementById("clear-search").style.display = "none";
    
    // Update Tags Display
    updateSymptomTags();
}

// Remove tag pill from selected array
function removeSymptom(sym) {
    selectedSymptoms.delete(sym);
    updateSymptomTags();
}

// Sync tag count badges and tags DOM nodes
function updateSymptomTags() {
    const container = document.getElementById("selected-tags-container");
    const countBadge = document.getElementById("selected-count");
    
    countBadge.innerText = selectedSymptoms.size;
    container.innerHTML = "";
    
    if (selectedSymptoms.size === 0) {
        container.innerHTML = `<div class="empty-state">No symptoms selected yet. Use the search box above.</div>`;
        return;
    }
    
    selectedSymptoms.forEach(sym => {
        const tag = document.createElement("div");
        tag.className = "tag";
        tag.innerHTML = `
            ${formatSymptomDisplay(sym)}
            <button class="remove-btn" onclick="removeSymptom('${sym}')">&times;</button>
        `;
        container.appendChild(tag);
    });
}

// Clear search input text manually
function clearSearchInput() {
    const input = document.getElementById("symptom-search");
    input.value = "";
    document.getElementById("symptoms-dropdown").style.display = "none";
    document.getElementById("clear-search").style.display = "none";
    input.focus();
}

// Reset entire symptom state
function resetSymptoms() {
    selectedSymptoms.clear();
    updateSymptomTags();
    
    // Hide prediction result panels
    document.getElementById("results-content").style.display = "none";
    document.getElementById("results-loading").style.display = "none";
    document.getElementById("results-placeholder").style.display = "flex";
}

// ==========================================================================
// Semantic Search Helpers
// ==========================================================================

// Apply a search suggestion chip directly to query bar
function applySearchSuggestion(queryText) {
    const input = document.getElementById("semantic-search-input");
    input.value = queryText;
    runSemanticSearch();
}

// Trigger search when pressing Enter key
function handleSearchKeyDown(event) {
    if (event.key === "Enter") {
        runSemanticSearch();
    }
}

// ==========================================================================
// Catalog Grid Search Filtering
// ==========================================================================

// Filter catalog grid cards alphabetically on typing
function filterCatalogGrid() {
    const query = document.getElementById("catalog-filter-input").value.trim().toLowerCase();
    
    const filtered = catalogData.filter(item => 
        item.disease.toLowerCase().includes(query) || 
        item.recommended.toLowerCase().includes(query) ||
        item.avoid.toLowerCase().includes(query)
    );
    
    renderCatalogGrid(filtered);
}

// ==========================================================================
// Tab Switching Management
// ==========================================================================
function switchTab(tabName) {
    // 1. Manage Active Tabs buttons css
    document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
    document.getElementById(`tab-${tabName}-btn`).classList.add("active");
    
    // 2. Hide all tab panels
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
    
    // 3. Show requested tab
    document.getElementById(`tab-${tabName}`).classList.add("active");
    
    // 4. Update Header Page Title
    const pageTitle = document.getElementById("page-title");
    if (tabName === 'predict') {
        pageTitle.innerText = "Symptom-based Nutritional Advisor";
    } else if (tabName === 'search') {
        pageTitle.innerText = "Semantic Dietary Search Platform";
    } else if (tabName === 'catalog') {
        pageTitle.innerText = "Nutritional Knowledge Catalog";
        // Proactively fetch catalog data
        fetchCatalogData();
    }
}

// Helper to show small notifications (fallback alert for errors)
function showErrorToast(msg) {
    alert(msg);
}
